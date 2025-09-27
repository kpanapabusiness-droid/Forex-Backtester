import pytest

from scripts.batch_sweeper import _sanitize_indicators


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            {"c1": "c1_supertrend", "c2": False, "baseline": False, "volume": False, "exit": False},
            {"c1": "c1_supertrend", "c2": None, "baseline": None, "volume": None, "exit": None},
        ),
        (
            {
                "c1": "c1_twiggs_money_flow",
                "c2": "false",
                "baseline": "none",
                "volume": "null",
                "exit": 0,
            },
            {
                "c1": "c1_twiggs_money_flow",
                "c2": None,
                "baseline": None,
                "volume": None,
                "exit": None,
            },
        ),
        (
            {"c1": "c1_aroon", "c2": None, "baseline": None, "volume": None, "exit": None},
            {"c1": "c1_aroon", "c2": None, "baseline": None, "volume": None, "exit": None},
        ),
        (
            {"c1": False, "c2": False, "baseline": False, "volume": False, "exit": False},
            {
                "c1": "c1_twiggs_money_flow",
                "c2": None,
                "baseline": None,
                "volume": None,
                "exit": None,
            },
        ),
    ],
)
def test_sanitize_indicators(raw, expected):
    assert _sanitize_indicators(raw) == expected
