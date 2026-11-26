"""API tests. They build a tiny model inside the test: no CSV, no model file."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app import main
from src.cinema.cleaning import CATEGORICAL, FEATURES, NUMERIC

GOOD = {"ticket_price_rs": 300, "days_since_release": 10, "star_rating": 4.0,
        "show_hour": 19, "screen_seats": 150, "distance_to_metro_km": 3.0,
        "language": "english", "location": "Mall"}


@pytest.fixture()
def client(monkeypatch):
    df = pd.DataFrame({
        "ticket_price_rs": [200, 300, 400, 250], "days_since_release": [1, 5, 9, 3],
        "star_rating": [3.5, 4.0, 4.5, 3.0], "show_hour": [12, 15, 19, 21],
        "screen_seats": [100, 150, 200, 120], "distance_to_metro_km": [1.0, 3.0, 5.0, 7.0],
        "language": ["english", "hindi", "regional", "english"],
        "location": ["Mall", "Highway", "City Centre", "Mall"]})
    y = [80, 60, 40, 30]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                             remainder="passthrough")
    tiny = Pipeline([("prep", prep), ("model", LinearRegression())]).fit(df[FEATURES], y)
    monkeypatch.setattr(main, "_model", tiny)
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert 0 <= r.json()["prediction"] <= 100


def test_predict_normalises_language(client):
    r = client.post("/predict", json={**GOOD, "language": " ENG. "})
    assert r.status_code == 200


def test_rejects_price_below_50(client):
    assert client.post("/predict", json={**GOOD, "ticket_price_rs": 49}).status_code == 422


def test_rejects_distance_above_8_5(client):
    assert client.post("/predict", json={**GOOD, "distance_to_metro_km": 8.6}).status_code == 422


def test_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "location"}
    assert client.post("/predict", json=bad).status_code == 422