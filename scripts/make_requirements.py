"""Writes requirements.txt with the EXACT versions installed in your venv.

Run:  python scripts/make_requirements.py
"""
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "cloudpickle",
            "mlflow", "pyyaml", "fastapi", "uvicorn", "pydantic", "httpx", "pytest"]

lines = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print("NOT INSTALLED:", name)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))