"""Test that an approved OPEN produces exactly 2 orders: TP1 and RUNNER."""


import pandas as pd

from live.reporting import write_orders_csv


def test_approved_open_produces_two_orders(tmp_path):
    run_id = "test-run"
    orders = [
        {
            "date_time": "2024-02-02 00:00:00",
            "symbol": "EURUSD",
            "direction": "long",
            "risk_pct": 0.25,
            "sl_price": 1.08,
            "tp_price": 1.10,
            "tag": "TP1",
            "run_id": run_id,
        },
        {
            "date_time": "2024-02-02 00:00:00",
            "symbol": "EURUSD",
            "direction": "long",
            "risk_pct": 0.25,
            "sl_price": 1.08,
            "tp_price": "",
            "tag": "RUNNER",
            "run_id": run_id,
        },
    ]
    write_orders_csv(tmp_path, orders, run_id)
    path = tmp_path / "orders.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 2
    tags = df["tag"].tolist()
    assert "TP1" in tags
    assert "RUNNER" in tags
    tp1_row = df[df["tag"] == "TP1"].iloc[0]
    assert tp1_row["tp_price"] != "" and float(tp1_row["tp_price"]) == 1.10
    runner_row = df[df["tag"] == "RUNNER"].iloc[0]
    assert runner_row["tp_price"] == "" or pd.isna(runner_row["tp_price"])
