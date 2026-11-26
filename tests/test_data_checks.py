"""Data-check tests on tiny data built inside the test."""
import pandas as pd

from src.cinema.data_checks import check_duplicates, check_sentinel, check_unit_mixup


def test_duplicates_counts_extra_copies():
    df = pd.DataFrame({"record_id": [1, 2, 2, 2], "x": [1, 2, 2, 2]})
    assert check_duplicates(df)["count"] == 2


def test_sentinel_found():
    df = pd.DataFrame({"days": [1, 2, 3, 4, 5, 9999, 9999, 9999, 9999, 9999]})
    res = check_sentinel(df, ["days"], [9999], min_count=5)
    assert res["present"] and res["value"] == 9999 and res["count"] == 5


def test_unit_mixup_found():
    df = pd.DataFrame({"loc": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                       "price": [300, 310, 290, 305, 295] * 2 + [30000, 31000, 29000, 30500, 29500]})
    res = check_unit_mixup(df, ["price"], "loc", [10, 100, 1000])
    assert res["present"] and res["group_value"] == "C" and res["factor"] == 100