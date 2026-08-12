from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from industry_momentum.data import (
    FILE_NAMES,
    download_datasets,
    load_industry_returns,
    stable_complete_start,
)


class DataTests(unittest.TestCase):
    def test_industry_parser_reads_only_monthly_value_weighted_table(self) -> None:
        names = [f"I{number:02d}" for number in range(49)]
        header = "," + ",".join(names)
        row_one = "202001," + ",".join(["1.00"] * 48 + ["-99.99"])
        row_two = "202002," + ",".join(["2.00"] * 49)
        payload = "\n".join(
            [
                "metadata",
                "Average Value Weighted Returns -- Monthly",
                header,
                row_one,
                row_two,
                "",
                "Average Equal Weighted Returns -- Monthly",
                header,
                row_two,
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("fixture.csv", payload)
            frame = load_industry_returns(path)

        self.assertEqual(frame.shape, (2, 49))
        self.assertAlmostEqual(frame.iloc[1, 0], 0.02)
        self.assertTrue(np.isnan(frame.iloc[0, -1]))
        self.assertEqual(frame.index[0], pd.Timestamp("2020-01-31"))

    def test_stable_complete_start_follows_last_missing_month(self) -> None:
        index = pd.date_range("2020-01-31", periods=5, freq="ME")
        frame = pd.DataFrame({"a": [1.0, np.nan, 2.0, 3.0, 4.0]}, index=index)
        self.assertEqual(stable_complete_start(frame), pd.Timestamp("2020-03-31"))

    def test_cached_archives_preserve_original_download_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            sources = {}
            for key, file_name in FILE_NAMES.items():
                (raw_dir / file_name).write_bytes(b"PK-cached-fixture")
                sources[key] = {"last_modified": f"original-{key}"}
            original_timestamp = "2026-08-12T20:00:00+00:00"
            (raw_dir / "manifest.json").write_text(
                json.dumps(
                    {"downloaded_at_utc": original_timestamp, "sources": sources}
                ),
                encoding="utf-8",
            )
            manifest = download_datasets(raw_dir)

        self.assertEqual(manifest["downloaded_at_utc"], original_timestamp)
        for key in FILE_NAMES:
            self.assertEqual(
                manifest["sources"][key]["last_modified"], f"original-{key}"
            )


if __name__ == "__main__":
    unittest.main()
