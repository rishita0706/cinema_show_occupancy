"""Small helper to read params.yaml (used by every module)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str = "params.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)