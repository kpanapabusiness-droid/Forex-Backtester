import pandas as pd


def _fake_trades(n=5):
    df = pd.DataFrame(
        {
            "win": [True, False, True, False, True],
            "scratch": [False, False, False, True, False],
            "pnl": [1.0, -0.5, 0.8, 0.0, 1.2],
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        }
    )
    return df


def test_boolean_indexing_uses_bitwise_ops():
    trades_df = _fake_trades()
    # This should work without raising "truth value of a Series is ambiguous"
    non_scratch = trades_df.loc[~trades_df["scratch"]]
    assert not non_scratch.empty
    losses = non_scratch.loc[~non_scratch["win"], "pnl"]
    assert (losses <= 0).all()
