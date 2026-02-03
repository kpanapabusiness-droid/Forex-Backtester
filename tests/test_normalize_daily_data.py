from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.normalize_daily_data import normalize_daily_file


def test_normalize_ftmo_daily_file(tmp_path: Path) -> None:
    # Create a FTMO-style tab-delimited file
    path = tmp_path / "EURUSD.csv"
    content = (
        "<DATE>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2019.01.02\t1.1000\t1.1100\t1.0900\t1.1050\t100\t10\t12\n"
        "2019.01.03\t1.1050\t1.1150\t1.0950\t1.1100\t200\t20\t15\n"
    )
    path.write_text(content, encoding="utf-8")

    converted, reason = normalize_daily_file(path)
    assert converted is True
    assert reason == "converted"

    text = path.read_text(encoding="utf-8")
    # Header should be exactly the legacy schema, comma-separated
    first_line = text.splitlines()[0]
    assert first_line == "time,open,high,low,close,tick_volume,spread,real_volume"

    # Parse with pandas using default (comma) separator
    df = pd.read_csv(path)
    assert df.columns.tolist() == [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]

    # Dates must be in YYYY-MM-DD format
    assert df["time"].iloc[0] == "2019-01-02"
    assert df["time"].iloc[1] == "2019-01-03"

