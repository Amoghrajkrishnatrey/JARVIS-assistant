"""
tools/eda.py
Exploratory data analysis tool: shape, dtypes, missing values, duplicate
rows, and summary statistics for a local CSV/Parquet file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import settings


def _resolve_path(filename: str) -> Path:
    p = Path(filename)
    if p.exists():
        return p
    return settings.data_dir / filename


def load_and_profile(filename: str, sample_rows: int = 5) -> str:
    """Load a CSV/Parquet file and return a text profile covering shape,
    dtypes, missing values, duplicate rows, and numeric summary stats."""
    path = _resolve_path(filename)
    if not path.exists():
        return f"File not found: {filename} (looked in current directory and {settings.data_dir})"

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return f"Unsupported file type for EDA: {path.suffix} (use .csv or .parquet)"
    except Exception as exc:
        return f"Failed to load {filename}: {exc}"

    missing = df.isna().sum()
    missing_pct = (missing / max(len(df), 1) * 100).round(2)
    missing_lines = [
        f"  - {col}: {missing[col]} missing ({missing_pct[col]}%)"
        for col in df.columns if missing[col] > 0
    ]
    missing_report = "\n".join(missing_lines) or "  None"

    dtypes_report = "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())

    numeric_df = df.select_dtypes(include="number")
    numeric_stats = numeric_df.describe().round(3).to_string() if not numeric_df.empty else "  No numeric columns."

    duplicate_count = int(df.duplicated().sum())

    return (
        f"=== EDA Profile: {path.name} ===\n"
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Data types:\n{dtypes_report}\n\n"
        f"Missing values:\n{missing_report}\n\n"
        f"Duplicate rows: {duplicate_count}\n\n"
        f"Numeric summary statistics:\n{numeric_stats}\n\n"
        f"Sample rows:\n{df.head(sample_rows).to_string()}"
    )
