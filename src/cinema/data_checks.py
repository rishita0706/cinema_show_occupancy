"""Finds the seven kinds of data problem in train.csv.

Run:  python -m src.cinema.data_checks
Every number printed here is what goes into reports/findings.yaml.
"""
import json
import re

import numpy as np
import pandas as pd

from src.cinema.config import ROOT, load_params


params = load_params()

# 1. leak
def check_leaking_column(train, holdout, target, threshold):
    """A column that is missing from the holdout file AND tracks the target."""
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    result = {"present": False, "column": None,
              "columns_missing_from_holdout": missing, "candidates": {},
              "corr_of_every_numeric_column_with_target": {}}
    for col in train.select_dtypes("number").columns:
        if col != target:
            result["corr_of_every_numeric_column_with_target"][col] = round(
                float(train[col].corr(train[target])), 3)
    for col in missing:
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = float(train[col].corr(train[target]))
            result["candidates"][col] = round(corr, 4)
            if abs(corr) >= threshold:
                result["present"] = True
                result["column"] = col
    return result


# 2. unit mixup
def check_unit_mixup(train, num_cols, group_col, factors):
    """One group value has a column about 10/12/60/100/1000x bigger or smaller."""
    result = {"present": False, "column": None, "group_value": None,
              "factor": None, "ratios": {}}
    best = None
    for col in num_cols:
        medians = train.groupby(group_col)[col].median()
        for grp, med in medians.items():
            others = train.loc[train[group_col] != grp, col].median()
            if others == 0 or med == 0:
                continue
            ratio = med / others
            result["ratios"][f"{col}|{grp}"] = round(float(ratio), 3)
            for f in factors:
                for r, fac in ((ratio, f), (1 / ratio, f)):
                    # relative closeness of the ratio to an exact factor
                    err = abs(r - fac) / fac
                    if err < 0.25 and (best is None or err < best[0]):
                        best = (err, col, grp, fac)
    if best:
        _, col, grp, fac = best
        result.update(present=True, column=col, group_value=grp, factor=fac)
    return result


# 3. sentinel
def check_sentinel(train, num_cols, candidates, min_count):
    """A special number typed instead of a blank, repeated many times."""
    result = {"present": False, "column": None, "value": None, "count": None,
              "all_hits": []}
    for col in num_cols:
        vc = train[col].value_counts()
        for val in candidates:
            if val in vc.index and vc[val] >= min_count:
                # a sentinel must be impossible for the column: look at the rest
                rest = train.loc[train[col] != val, col]
                if val < rest.min() or val > rest.max():
                    result["all_hits"].append(
                        {"column": col, "value": val, "count": int(vc[val])})
    if result["all_hits"]:
        top = max(result["all_hits"], key=lambda h: h["count"])
        result.update(present=True, column=top["column"], value=top["value"],
                      count=top["count"])
    return result


# 4. duplicates
def check_duplicates(train):
    """Number of EXTRA copies (rows you would delete)."""
    n = int(train.duplicated().sum())
    n_ignoring_id = int(train.drop(columns=["record_id"]).duplicated().sum())
    return {"present": n > 0, "count": n if n > 0 else None,
            "extra_copies_ignoring_record_id": n_ignoring_id}


# 5. stuck sensor
def check_stuck_sensor(train, num_cols, date_col, target, min_run):
    """A column repeating the same value for a long run (sorted by date)."""
    df = train.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, "record_id"]).reset_index(drop=True)
    result = {"present": False, "column": None, "start_date": None,
              "end_date": None, "runs": []}
    for col in num_cols:
        s = df[col]
        run_id = (s != s.shift()).cumsum()
        sizes = s.groupby(run_id).transform("size")
        longest = sizes.max()
        result.setdefault("longest_run_per_column", {})[col] = int(longest)
        if longest >= min_run:
            idx = df.index[sizes == longest]
            block = df.loc[idx]
            result["runs"].append({
                "column": col, "value": float(block[col].iloc[0]),
                "rows": int(longest),
                "start_date": block[date_col].min().strftime("%Y-%m-%d"),
                "end_date": block[date_col].max().strftime("%Y-%m-%d")})
    if result["runs"]:
        top = max(result["runs"], key=lambda r: r["rows"])
        result.update(present=True, column=top["column"],
                      start_date=top["start_date"], end_date=top["end_date"])
    return result


# 6. target typos
def check_target_typos(train, target):
    """Targets typed 10 times too big: a small cluster far above a big empty gap."""
    s = train[target].dropna().sort_values().reset_index(drop=True)
    upper = s[s >= s.median()]                 # typos can only be on the high side
    gaps = upper.diff()
    i = int(gaps.idxmax())
    gap = float(gaps.loc[i])
    cut = float(s.loc[i - 1])
    count = int((train[target] > cut).sum())
    above_100 = int((train[target] > 100).sum())   # a percent cannot exceed 100
    present = gap > 15 and 0 < count <= 0.1 * len(train)
    return {"present": present, "count": count if present else None,
            "largest_gap_in_upper_half": round(gap, 2),
            "max_value": float(s.max()), "rows_above_100": above_100}


# 7. spelling variants
def _norm(x):
    return re.sub(r"[^a-z]", "", str(x).lower())


def check_spelling_variants(train, later, cat_cols):
    result = {"present": False, "column": None, "true_levels": None,
              "details": {}}
    for col in cat_cols:
        raw = train[col].value_counts(dropna=False).to_dict()
        clean_levels = sorted(later[col].dropna().unique())
        result["details"][col] = {"raw_levels": raw,
                                  "clean_levels_from_later": clean_levels}
        if len(raw) > len(clean_levels):
            result.update(present=True, column=col, true_levels=len(clean_levels))
    return result


# main
def run_all():
    p = load_params()
    train = pd.read_csv(ROOT / p["paths"]["train"])
    later = pd.read_csv(ROOT / p["paths"]["later"])
    holdout = pd.read_csv(ROOT / p["paths"]["holdout"])
    target, grp, dt = p["target"], p["group_column"], p["date_column"]
    nums, cats = p["numeric_features"], p["categorical_features"]
    c = p["data_checks"]

    print(f"train rows: {len(train)} | later rows: {len(later)} | holdout rows: {len(holdout)}")
    print("train null counts:\n", train.isna().sum().to_string(), "\n")

    # unit-mixup and sentinel checks need clean group labels
    train_norm = train.copy()
    train_norm[grp] = train_norm[grp].astype(str).str.strip()

    report = {
        "leaking_column": check_leaking_column(train, holdout, target, c["leak_corr_threshold"]),
        "unit_mixup": check_unit_mixup(train_norm, nums, grp, c["unit_factors"]),
        "sentinel_value": check_sentinel(train, nums, c["sentinel_candidates"], c["min_sentinel_count"]),
        "duplicate_rows": check_duplicates(train),
        "stuck_sensor": check_stuck_sensor(train, nums, dt, target, c["stuck_min_run"]),
        "target_typos": check_target_typos(train, target),
        "spelling_variants": check_spelling_variants(train, later, cats),
    }
    for name, res in report.items():
        print(f"=== {name} ===")
        print(json.dumps(res, indent=2, default=str))
        print()

    out = ROOT / p["paths"]["data_issues_report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved full report to {out}")
    return report


if __name__ == "__main__":
    run_all()