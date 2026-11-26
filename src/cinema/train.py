"""Train v1 with MLflow tracking, register the best run, give it @champion.

Run ONCE:  python -m src.cinema.train
(To start again, delete mlflow.db and the mlartifacts folder first.)
"""
import hashlib
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cinema.cleaning import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_train
from src.cinema.config import ROOT, load_params

P = load_params()
SEED = P["training"]["random_state"]

# the 4 settings we try (every one becomes one MLflow run)
SETTINGS = [
    {"run_name": "ridge_alpha1", "model": "ridge", "alpha": 1.0},
    {"run_name": "rf_100_depth_none", "model": "random_forest",
     "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"run_name": "rf_300_depth8_leaf3", "model": "random_forest",
     "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3},
    {"run_name": "gbr_300_lr005_depth3", "model": "gradient_boosting",
     "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
]


def data_sha256() -> str:
    """SHA-256 of the ORIGINAL train.csv (the file DVC tracks)."""
    with open(ROOT / P["paths"]["train"], "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(setting: dict) -> Pipeline:
    """Raw 8 columns in -> prediction out (encoding is inside the model)."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    kind = setting["model"]
    if kind == "ridge":
        model = Ridge(alpha=setting["alpha"])
    elif kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=setting["n_estimators"], max_depth=setting["max_depth"],
            min_samples_leaf=setting["min_samples_leaf"], random_state=SEED, n_jobs=-1)
    elif kind == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=setting["n_estimators"], learning_rate=setting["learning_rate"],
            max_depth=setting["max_depth"], random_state=SEED)
    else:
        raise ValueError(f"unknown model {kind}")
    return Pipeline([("prep", prep), ("model", model)])


def get_xy(df):
    X = df[FEATURES].copy()
    X[NUMERIC] = X[NUMERIC].astype(float)
    return X, df[TARGET]


def setup_mlflow():
    cfg = P["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    name = cfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / cfg["artifact_dir"]).as_uri())
    mlflow.set_experiment(name)


def log_sklearn_model(pipe, X):
    """Logs the model; cloudpickle avoids the 'untrusted types' error of MLflow 3."""
    sig = infer_signature(X.head(5), pipe.predict(X.head(5)))
    try:
        return mlflow.sklearn.log_model(pipe, name="model", signature=sig,
                                        serialization_format="cloudpickle")
    except TypeError:       # older MLflow versions use artifact_path
        return mlflow.sklearn.log_model(pipe, artifact_path="model", signature=sig,
                                        serialization_format="cloudpickle")


def run_setting(setting, X, y, sha):
    """One MLflow run: cross-validated MAE, then fit on all of train and log."""
    pipe = build_pipeline(setting)
    cv = KFold(n_splits=P["training"]["cv_folds"], shuffle=True, random_state=SEED)
    cv_mae = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()

    with mlflow.start_run(run_name=setting["run_name"]) as run:
        mlflow.log_param("data_sha256", sha)
        mlflow.log_param("model_version_label", "v1")
        mlflow.log_param("n_train_rows", len(X))
        mlflow.log_params({k: v for k, v in setting.items() if k != "run_name"})
        pipe.fit(X, y)
        mlflow.log_metric("mae_cv", float(cv_mae))
        mlflow.log_metric("mae_train", float(mean_absolute_error(y, pipe.predict(X))))
        info = log_sklearn_model(pipe, X)
        return {"run_id": run.info.run_id, "run_name": setting["run_name"],
                "mae_cv": float(cv_mae), "model_uri": info.model_uri}


def register_champion(best: dict) -> str:
    name = P["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(best["model_uri"], name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)
    return mv.version


def export_champion():
    """Copy the @champion model to models/model.joblib (the API loads this)."""
    name = P["mlflow"]["registered_model_name"]
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    out = ROOT / P["mlflow"]["model_export_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    return out


def main(force=False):
    setup_mlflow()
    name = P["mlflow"]["registered_model_name"]
    if MlflowClient().search_registered_models(filter_string=f"name='{name}'") and not force:
        sys.exit("Model already registered. Delete mlflow.db and mlartifacts/ to start again.")

    X, y = get_xy(load_train())
    sha = data_sha256()
    print("data_sha256:", sha)
    print("training rows:", len(X))

    results = [run_setting(s, X, y, sha) for s in SETTINGS]
    for r in results:
        print(f"  {r['run_name']:<24} mae_cv = {r['mae_cv']:.3f}")

    best = min(results, key=lambda r: r["mae_cv"])
    version = register_champion(best)
    path = export_champion()

    print("\n--- copy these into reports/findings.yaml (tracking:) ---")
    print("experiment_name      :", P["mlflow"]["experiment_name"])
    print("best_run_id          :", best["run_id"])
    print("registered_model_name:", name)
    print("champion_version     :", version)
    print(f"\nbest run: {best['run_name']} (mae_cv {best['mae_cv']:.3f})")
    print("model saved to:", path)


if __name__ == "__main__":
    main(force="--force" in sys.argv)