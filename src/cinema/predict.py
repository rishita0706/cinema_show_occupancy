"""Predict data/raw/holdout_features.csv with the @champion model.

Run:  python -m src.cinema.predict
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.cinema import train as tr
from src.cinema.cleaning import FEATURES, load_holdout
from src.cinema.config import ROOT, load_params

P = load_params()
MODEL_NAME = P["mlflow"]["registered_model_name"]
OUT = ROOT / "predictions" / "holdout_predictions.csv"


def main():
    mlflow.set_tracking_uri(P["mlflow"]["tracking_uri"])
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    holdout = load_holdout()
    X = holdout[FEATURES].copy()
    X[P["numeric_features"]] = X[P["numeric_features"]].astype(float)
    pred = np.clip(model.predict(X), 0, 100)

    out = pd.DataFrame({"record_id": holdout["record_id"], "prediction": pred.round(4)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("champion version used:", version)
    print("rows written:", len(out), "| columns:", list(out.columns))
    print(out.head().to_string(index=False))
    print("prediction min/mean/max:", out.prediction.min(), round(out.prediction.mean(), 2),
          out.prediction.max())
    print("saved to", OUT)


if __name__ == "__main__":
    main()