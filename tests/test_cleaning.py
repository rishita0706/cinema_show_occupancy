"""Cleaning tests on tiny data built inside the test."""
import numpy as np
import pandas as pd

from src.cinema.cleaning import clean_language, clean_train


def make_df():
    return pd.DataFrame({
        "record_id": [1, 2, 3, 3, 4],
        "date": ["2026-01-01"] * 5,
        "ticket_price_rs": [300.0, 30000.0, 280.0, 280.0, 310.0],
        "days_since_release": [5, 9999, 7, 7, 12],
        "star_rating": [4.0, 3.5, 4.2, 4.2, 3.9],
        "show_hour": [12, 15, 18, 18, 21],
        "screen_seats": [100, 120, 90, 90, 110],
        "distance_to_metro_km": [2.0, 4.0, 3.0, 3.0, 5.0],
        "language": ["English", "hin.", " REGIONAL", " REGIONAL", "english "],
        "location": ["Mall", "Highway", "City Centre", "City Centre", "Mall"],
        "occupancy_pct": [60.0, 50.0, 70.0, 70.0, 40.0]})


def test_clean_language_variants():
    assert clean_language(" ENGLISH ") == "english"
    assert clean_language("hin.") == "hindi"
    assert clean_language("reg.") == "regional"


def test_clean_train_fixes_everything():
    out = clean_train(make_df())
    assert len(out) == 4                                     # duplicate removed
    assert set(out["language"]) == {"english", "hindi", "regional"}
    assert out.loc[out["location"] == "Highway", "ticket_price_rs"].iloc[0] == 300.0
    assert not (out["days_since_release"] == 9999).any()
    assert not out["days_since_release"].isna().any()