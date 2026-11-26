"""Self-check before you submit.   Run:  python scripts/check_submission.py

It checks the things the marking tool checks (handbook section 11). It cannot
check GitHub (public link, green tick) or the form: do those yourself.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cinema.config import load_params  # noqa: E402

P = load_params()
results = []


def check(ok, label, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + label + (f"   -> {detail}" if detail and not ok else ""))


def has_none(d):
    if isinstance(d, dict):
        return any(has_none(v) for v in d.values())
    return d is None


# ---- files that must exist -------------------------------------------------
for f in ["README.md", "requirements.txt", "params.yaml", "mlflow.db", "app/main.py",
          "models/model.joblib", "Dockerfile", ".dockerignore", ".github/workflows/ci.yml",
          "k8s/deployment.yaml", "k8s/service.yaml", "reports/findings.yaml",
          "predictions/holdout_predictions.csv", "data/raw/holdout_features.csv",
          "data/raw/train.csv.dvc", "data/raw/later.csv.dvc", ".dvc/config"]:
    check((ROOT / f).exists(), f"file exists: {f}")

# ---- README ---------------------------------------------------------------
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
group = (ROOT / "GROUP.md").read_text(encoding="utf-8") if (ROOT / "GROUP.md").exists() else ""
ids = re.findall(r"S\d{2}[A-Z]{4}\d{4}", group)
check("G215" in readme and "MP152" in readme, "README has group id and problem id")
check(bool(ids) and all(i in readme for i in ids), f"README has all {len(ids)} enrolment numbers")

# ---- requirements pinned --------------------------------------------------
req = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip()]
check(bool(req) and all("==" in l for l in req), "every requirement is pinned with ==")

# ---- CI starts with on: -----------------------------------------------------
ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8").lstrip()
check(ci.startswith("on:"), "ci.yml starts with 'on:'")

# ---- findings.yaml -----------------------------------------------------------
f = yaml.safe_load((ROOT / "reports/findings.yaml").read_text(encoding="utf-8"))
for name, block in f["data_issues"].items():
    if block["present"]:
        check(not has_none(block), f"findings: {name} fully filled")
    else:
        check(all(v is None for k, v in block.items() if k != "present"),
              f"findings: {name} absent -> other values are null")
check(not has_none(f["tracking"]), "findings: tracking has no null")
check(not has_none(f["monitoring"]), "findings: monitoring has no null")

# ---- findings agree with the generated reports -----------------------------
rep_path = ROOT / "reports/monitoring_report.json"
if rep_path.exists():
    rep = json.loads(rep_path.read_text())
    m = f["monitoring"]
    close = lambda a, b: a is not None and abs(float(a) - float(b)) < 0.01
    check(all(close(m["psi"][c], v) for c, v in rep["psi"].items()), "findings PSI = monitoring_report")
    check(m["most_drifted_feature"] == rep["most_drifted_feature"], "most_drifted_feature matches")
    check(all(close(m["mae_by_group"].get(g), v) for g, v in rep["mae_by_group"].items()),
          "mae_by_group matches")
    check(m["worst_served_group"] == rep["worst_served_group"], "worst_served_group matches")
    check(close(m["mae_v1_on_check"], rep["mae_v1_on_check"]) and
          close(m["mae_v2_on_check"], rep["mae_v2_on_check"]), "mae_v1/mae_v2 on check match")
    check(m["decision"] == rep["decision"], "decision matches")
    check(f["tracking"]["champion_version"] == rep["champion_version"], "champion_version matches")
else:
    check(False, "reports/monitoring_report.json exists (run monitor.py)")

# ---- predictions ------------------------------------------------------------
hold = pd.read_csv(ROOT / "data/raw/holdout_features.csv")
pred = pd.read_csv(ROOT / "predictions/holdout_predictions.csv")
check(list(pred.columns) == ["record_id", "prediction"], "predictions columns are record_id,prediction")
check(set(pred.record_id) == set(hold.record_id) and len(pred) == len(hold),
      f"predictions cover all {len(hold)} holdout rows", f"{len(pred)} rows")
check(pred.prediction.notna().all(), "no empty predictions")

# ---- DVC: train.csv.dvc hash = original file --------------------------------
dvc = ROOT / "data/raw/train.csv.dvc"
csv = ROOT / P["paths"]["train"]
if dvc.exists() and csv.exists():
    md5 = hashlib.md5(csv.read_bytes()).hexdigest()
    check(md5 in dvc.read_text(), "train.csv.dvc hash matches data/raw/train.csv")
# git must not track the big CSVs
gi = ROOT / ".git"
if gi.exists():
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True,
                             text=True).stdout.split()
    check("data/raw/train.csv" not in tracked and "data/raw/later.csv" not in tracked,
          "train.csv / later.csv are NOT committed to git")
    check("mlflow.db" in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.split(), "mlflow.db is committed")

# ---- MLflow ----------------------------------------------------------------
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

mlflow.set_tracking_uri(P["mlflow"]["tracking_uri"])
client = MlflowClient()
exp = client.get_experiment_by_name(P["mlflow"]["experiment_name"])
runs = client.search_runs([exp.experiment_id]) if exp else []
check(len(runs) >= 3, f"MLflow has 3+ runs ({len(runs)})")
check(all("data_sha256" in r.data.params for r in runs), "every run has data_sha256")
check(all(any("mae" in k.lower() for k in r.data.metrics) for r in runs), "every run has an MAE metric")
name = P["mlflow"]["registered_model_name"]
versions = client.search_model_versions(f"name='{name}'")
check(len(versions) >= 2, f"registry has 2+ versions ({len(versions)})")
try:
    champ = client.get_model_version_by_alias(name, "champion")
    check(True, f"alias champion -> version {champ.version}")
    check(str(f["tracking"]["champion_version"]) == str(champ.version),
          "findings champion_version = registry champion")
    check(f["tracking"]["best_run_id"] in [r.info.run_id for r in runs], "best_run_id exists in MLflow")
except Exception as e:
    check(False, "alias champion is set", str(e))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)