"""Download and parse the public Kenneth French research datasets."""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DATA_URLS: Mapping[str, str] = {
    "industry_49": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "49_Industry_Portfolios_CSV.zip"
    ),
    "momentum_factor": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_CSV.zip"
    ),
    "ff3_factors": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_CSV.zip"
    ),
}

FILE_NAMES: Mapping[str, str] = {
    "industry_49": "49_Industry_Portfolios_CSV.zip",
    "momentum_factor": "F-F_Momentum_Factor_CSV.zip",
    "ff3_factors": "F-F_Research_Data_Factors_CSV.zip",
}

MONTHLY_ROW = re.compile(r"^\s*\d{6},")


@dataclass(frozen=True)
class DataBundle:
    """Parsed monthly returns and the manifest for their source archives."""

    industry_returns: pd.DataFrame
    factors: pd.DataFrame
    manifest: dict[str, object]
    stable_start: pd.Timestamp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "industry-momentum-oos/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
        last_modified = response.headers.get("Last-Modified")
    if not payload.startswith(b"PK"):
        raise ValueError(f"Expected a ZIP archive from {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return {
        "url": url,
        "file": destination.name,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "last_modified": last_modified,
    }


def download_datasets(raw_dir: Path, force: bool = False) -> dict[str, object]:
    """Download official source archives and write a checksum manifest.

    Source archives stay under ``data/raw`` and are excluded from Git. This
    avoids redistributing the copyrighted research data while retaining an
    auditable source URL and checksum for the exact research run.
    """

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    previous_manifest: dict[str, object] = {}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_sources = previous_manifest.get("sources", {})
    if not isinstance(previous_sources, dict):
        previous_sources = {}

    sources: dict[str, object] = {}
    downloaded_any = False
    for key, url in DATA_URLS.items():
        destination = raw_dir / FILE_NAMES[key]
        if force or not destination.exists():
            sources[key] = _download(url, destination)
            downloaded_any = True
        else:
            prior = previous_sources.get(key, {})
            if not isinstance(prior, dict):
                prior = {}
            sources[key] = {
                "url": url,
                "file": destination.name,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "last_modified": prior.get("last_modified"),
            }
    prior_downloaded_at = previous_manifest.get("downloaded_at_utc")
    manifest: dict[str, object] = {
        "downloaded_at_utc": (
            datetime.now(timezone.utc).isoformat()
            if downloaded_any or not isinstance(prior_downloaded_at, str)
            else prior_downloaded_at
        ),
        "sources": sources,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _read_archive_lines(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member in {path}, found {len(members)}")
        return archive.read(members[0]).decode("utf-8").splitlines()


def _parse_monthly_rows(lines: list[str], header_index: int) -> pd.DataFrame:
    end = header_index + 1
    while end < len(lines) and MONTHLY_ROW.match(lines[end]):
        end += 1
    if end == header_index + 1:
        raise ValueError("No monthly YYYYMM rows found after the CSV header")

    frame = pd.read_csv(io.StringIO("\n".join(lines[header_index:end])), skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    frame = frame.set_index("date").apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([-99.99, -999.0], np.nan) / 100.0
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Monthly index must be unique and increasing")
    return frame


def load_industry_returns(path: Path) -> pd.DataFrame:
    """Load monthly value-weighted returns for 49 industry portfolios."""

    lines = _read_archive_lines(path)
    title = next(
        index
        for index, line in enumerate(lines)
        if "Average Value Weighted Returns -- Monthly" in line
    )
    frame = _parse_monthly_rows(lines, title + 1)
    if frame.shape[1] != 49:
        raise ValueError(f"Expected 49 industry columns, found {frame.shape[1]}")
    return frame


def load_factor_returns(path: Path) -> pd.DataFrame:
    """Load the first (monthly) table from a French factor archive."""

    lines = _read_archive_lines(path)
    header = next(index for index, line in enumerate(lines) if line.startswith(","))
    return _parse_monthly_rows(lines, header)


def stable_complete_start(frame: pd.DataFrame) -> pd.Timestamp:
    """Return the month after the final row containing any missing value."""

    missing = frame.isna().any(axis=1)
    if not missing.any():
        return frame.index.min()
    last_incomplete = frame.index[missing][-1]
    later = frame.index[frame.index > last_incomplete]
    if later.empty:
        raise ValueError("No complete sample exists after the last missing row")
    start = later[0]
    if frame.loc[start:].isna().any().any():
        raise AssertionError("Stable sample still contains missing observations")
    return start


def prepare_data(raw_dir: Path, force_download: bool = False) -> DataBundle:
    """Download, parse, align, and validate all datasets used in the study."""

    manifest = download_datasets(raw_dir, force=force_download)
    industries = load_industry_returns(raw_dir / FILE_NAMES["industry_49"])
    momentum = load_factor_returns(raw_dir / FILE_NAMES["momentum_factor"])
    ff3 = load_factor_returns(raw_dir / FILE_NAMES["ff3_factors"])
    factors = ff3.join(momentum, how="inner")
    expected = {"Mkt-RF", "SMB", "HML", "RF", "Mom"}
    if not expected.issubset(factors.columns):
        raise ValueError(f"Factor columns differ from expected set: {list(factors.columns)}")
    start = stable_complete_start(industries)
    return DataBundle(
        industry_returns=industries.loc[start:].copy(),
        factors=factors,
        manifest=manifest,
        stable_start=start,
    )
