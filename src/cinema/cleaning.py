"""Cleaning code: fixes the data problems found by data_checks.py.

Run:  python -m src.cinema.cleaning     (prints a before/after summary)

Problems fixed (from findings):
  duplicate_rows     -> drop exact duplicate rows
  unit_mixup         -> Highway ticket_price_rs was typed x100, divide by 100
  sentinel_value     -> days_since_release == 9999 is a blank, fill with the median
  spelling_variants  -> language written 18 ways, map to english / hindi / regional
Not present (nothing to fix): leaking_column, stuck_sensor, target_typos.
"""
import numpy as np
import pandas as pd

from src.cinema.config import ROOT, load_params

PARAMS = load_params()
NUMERIC = PARAMS["numeric_features"]
CATEGORICAL = PARAMS["categorical_features"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = PARAMS["target"]

# values found by data_checks.py (kept in one place so the viva is easy)
UNIT_GROUP = "Highway"
UNIT_COLUMN = "ticket_price_rs"
UNIT_FACTOR = 100
SENTINEL_COLUMN = "days_since_release"
SENTINEL_VALUE = 9999


def clean_language(value: str) -> str:
    """' English', 'ENGLISH', 'eng.' -> 'english' (same for hindi, regional)."""
    v = str(value).strip().lower()
    if v.startswith("eng"):
        return "english"
    if v.startswith("hin"):
        return "hindi"
    if v.startswith("reg"):
        return "regional"
    return v


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the messy train.csv. Do NOT use this on later.csv (it is clean)."""
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)          # duplicate_rows

    df["language"] = df["language"].map(clean_language)       # spelling_variants
    df["location"] = df["location"].str.strip()

    fix = df["location"] == UNIT_GROUP                        # unit_mixup
    df.loc[fix, UNIT_COLUMN] = df.loc[fix, UNIT_COLUMN] / UNIT_FACTOR

    df[SENTINEL_COLUMN] = df[SENTINEL_COLUMN].replace(SENTINEL_VALUE, np.nan)
    df[SENTINEL_COLUMN] = df[SENTINEL_COLUMN].fillna(df[SENTINEL_COLUMN].median())
    return df


def clean_later(df: pd.DataFrame) -> pd.DataFrame:
    """later.csv and holdout are already clean; only tidy the text columns."""
    df = df.copy()
    df["language"] = df["language"].map(clean_language)
    df["location"] = df["location"].str.strip()
    return df


def load_train() -> pd.DataFrame:
    return clean_train(pd.read_csv(ROOT / PARAMS["paths"]["train"]))


def load_later() -> pd.DataFrame:
    return clean_later(pd.read_csv(ROOT / PARAMS["paths"]["later"]))


def load_holdout() -> pd.DataFrame:
    return clean_later(pd.read_csv(ROOT / PARAMS["paths"]["holdout"]))


if __name__ == "__main__":
    raw = pd.read_csv(ROOT / PARAMS["paths"]["train"])
    clean = clean_train(raw)
    print("rows before:", len(raw), "| after:", len(clean))
    print("language levels:", sorted(clean["language"].unique()))
    print("max days_since_release:", clean["days_since_release"].max())
    print("\nticket_price_rs by location (mean):")
    print(clean.groupby("location")["ticket_price_rs"].mean().round(1).to_string())
    print("\nnulls left:", int(clean[FEATURES + [TARGET]].isna().sum().sum()))