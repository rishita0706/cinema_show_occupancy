"""Monitoring and retraining (PROBLEM.md section 7).

Run ONCE, after train.py:   python -m src.cinema.monitor

1. PSI of every numeric column, cleaned train vs later.csv (10 bins)
2. v1's MAE on later.csv for each location
3. retrain v2 on train + first 70% of later (by date); compare v1 and v2 on last 30%
4. promotion rule: promote v2 only if its MAE is at least 10% lower than v1's
"""
import json
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

from src.cinema import train as tr
from src.cinema.cleaning import FEATURES, NUMERIC, TARGET, load_later, load_train
from src.cinema.config import ROOT, load_params

P = load_params()
MODEL_NAME = P["mlflow"]["registered_model_name"]
V1_VERSION = "1"
TRAIN_FRACTION = 0.70          # first 70% of later.csv goes into v2's training data
PROMOTION_GAIN = 0.10          # client rule: v2 must be at least 10% better
EPS = 1e-4                     # stops log(0) when a bin is empty


# ------------------------------------------------------------------ 1. PSI
def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. Bins come from the quantiles of the OLD data."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf            # catch values outside the old range
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_all(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    return {col: round(psi(train[col], later[col]), 4) for col in NUMERIC}


# ------------------------------------------------------------- helpers / MAE
def predict_clipped(model, df: pd.DataFrame) -> np.ndarray:
    X, _ = tr.get_xy(df)
    return np.clip(model.predict(X), 0, 100)


def mae(model, df: pd.DataFrame) -> float:
    return float(mean_absolute_error(df[TARGET], predict_clipped(model, df)))


def mae_by_group(model, df: pd.DataFrame, group: str) -> dict:
    return {g: round(mae(model, part), 4) for g, part in df.groupby(group)}


def setting_of_version(version: str) -> dict:
    """Find which of train.SETTINGS produced a registered model version."""
    mv = MlflowClient().get_model_version(MODEL_NAME, version)
    run_name = mlflow.get_run(mv.run_id).data.tags["mlflow.runName"]
    return next(s for s in tr.SETTINGS if s["run_name"] == run_name)


# --------------------------------------------------------------------- main
def main(force: bool = False):
    tr.setup_mlflow()
    client = MlflowClient()
    if len(client.search_model_versions(f"name='{MODEL_NAME}'")) > 1 and not force:
        sys.exit("v2 already exists. Delete mlflow.db and mlartifacts/, run train.py, "
                 "then run monitor.py again.")

    train = load_train()
    later = load_later().sort_values(["date", "record_id"]).reset_index(drop=True)
    v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{V1_VERSION}")

    # 1. PSI
    psi_values = psi_all(train, later)
    most_drifted = max(psi_values, key=psi_values.get)

    # 2. v1 MAE per location on all of later.csv
    group_mae = mae_by_group(v1, later, P["group_column"])
    worst_group = max(group_mae, key=group_mae.get)

    # 3. retrain v2 on train + first 70% of later; compare on the last 30%
    n_fit = int(round(len(later) * TRAIN_FRACTION))
    later_fit, later_check = later.iloc[:n_fit], later.iloc[n_fit:]
    v2_data = pd.concat([train, later_fit], ignore_index=True)
    X2, y2 = tr.get_xy(v2_data)
    setting = setting_of_version(V1_VERSION)          # same recipe as v1, new data
    v2 = tr.build_pipeline(setting).fit(X2, y2)

    mae_v1 = mae(v1, later_check)
    mae_v2 = mae(v2, later_check)
    gain = (mae_v1 - mae_v2) / mae_v1

    # 4. promotion rule
    promote = gain >= PROMOTION_GAIN
    decision = "promote" if promote else "keep"

    with mlflow.start_run(run_name=f"v2_{setting['run_name']}") as run:
        mlflow.log_param("data_sha256", tr.data_sha256())
        mlflow.log_param("model_version_label", "v2")
        mlflow.log_param("n_train_rows", len(v2_data))
        mlflow.log_params({k: v for k, v in setting.items() if k != "run_name"})
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        mlflow.log_metric("mae_gain_pct", gain * 100)
        info = tr.log_sklearn_model(v2, X2)
    mv2 = mlflow.register_model(info.model_uri, MODEL_NAME)

    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv2.version)
        tr.export_champion()                           # models/model.joblib = v2
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion").version

    report = {
        "psi": psi_values, "most_drifted_feature": most_drifted,
        "mae_by_group": group_mae, "worst_served_group": worst_group,
        "rows_check": len(later_check), "rows_v2_training": len(v2_data),
        "mae_v1_on_check": round(mae_v1, 4), "mae_v2_on_check": round(mae_v2, 4),
        "gain_pct": round(gain * 100, 2), "decision": decision,
        "champion_version": int(champion), "v2_run_id": run.info.run_id,
    }
    out = ROOT / P["paths"]["reports_dir"] / "monitoring_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("PSI per column (train vs later):")
    for col, v in psi_values.items():
        print(f"  {col:<22} {v:.4f}")
    print("most drifted feature :", most_drifted)
    print("\nv1 MAE per location on later.csv:")
    for g, v in group_mae.items():
        print(f"  {g:<12} {v:.4f}")
    print("worst served group   :", worst_group)
    print(f"\nlast 30% of later ({len(later_check)} rows):")
    print(f"  v1 MAE = {mae_v1:.4f} | v2 MAE = {mae_v2:.4f} | gain = {gain * 100:.2f}% "
          f"(need >= {PROMOTION_GAIN * 100:.0f}%)")
    print("decision             :", decision)
    print("champion_version     :", champion)
    print("\nfull report saved to", out)
    return report


if __name__ == "__main__":
    main(force="--force" in sys.argv)