"""
Data handler utilities for the CFD prediction system.

This module accepts CSV/Excel uploads, validates the expected schema,
normalizes numeric columns, and stores cleaned datasets plus upload metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

EXPECTED_COLUMNS_BASE = [
    "inlet_velocity",
    "temperature",
    "diameter",
    "valve_opening",
    "max_pressure",
]

OPTIONAL_COLUMNS = [
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "p",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
META_PATH = os.path.join(DATA_DIR, "meta.json")


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def _compute_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(hashed).hexdigest()


def _load_meta() -> Dict[str, Any]:
    _ensure_dirs()
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"uploads": [], "latest_csv": None}


def _save_meta(meta: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(META_PATH, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def validate_data(df: pd.DataFrame, strict: bool = False) -> Tuple[bool, str]:
    normalized = _normalize_columns(df)
    if normalized.empty:
        return False, "DataFrame is empty."

    missing = [column for column in EXPECTED_COLUMNS_BASE if column not in normalized.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    if strict:
        missing_optional = [column for column in OPTIONAL_COLUMNS if column not in normalized.columns]
        if missing_optional:
            return False, f"Missing optional columns in strict mode: {missing_optional}"

    numeric_check = normalized.copy()
    for column in EXPECTED_COLUMNS_BASE:
        numeric_check[column] = pd.to_numeric(numeric_check[column], errors="coerce")

    nan_counts = numeric_check[EXPECTED_COLUMNS_BASE].isna().sum()
    if nan_counts.any():
        bad_columns = nan_counts[nan_counts > 0].index.tolist()
        return False, f"Missing or non-numeric values found in required columns: {bad_columns}"

    return True, "Valid"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    processed = _normalize_columns(df)
    processed = processed.drop_duplicates()

    numeric_columns = [column for column in EXPECTED_COLUMNS_BASE + OPTIONAL_COLUMNS if column in processed.columns]
    for column in numeric_columns:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    processed = processed.dropna(subset=EXPECTED_COLUMNS_BASE)
    return processed.reset_index(drop=True)


def store_dataframe(df: pd.DataFrame, label: Optional[str] = None) -> str:
    _ensure_dirs()

    valid, message = validate_data(df)
    if not valid:
        raise ValueError(f"Validation failed: {message}")

    processed = preprocess_data(df)
    dataset_hash = _compute_hash(processed)
    output_name = f"dataset_{dataset_hash[:12]}.csv"
    output_path = os.path.join(DATA_DIR, output_name)
    processed.to_csv(output_path, index=False)

    meta = _load_meta()
    existing = next(
        (
            entry
            for entry in meta.get("uploads", [])
            if entry.get("hash") == dataset_hash and entry.get("path") == output_name
        ),
        None,
    )
    if existing is None:
        meta["uploads"].append(
            {
                "hash": dataset_hash,
                "label": label or output_name,
                "path": output_name,
                "rows": int(len(processed)),
                "columns": list(processed.columns),
            }
        )
    elif label:
        existing["label"] = label
    meta["latest_csv"] = output_name
    _save_meta(meta)
    return output_path


def upload_data(file_path: str, label: Optional[str] = None) -> str:
    _ensure_dirs()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".csv":
        df = pd.read_csv(file_path)
    elif extension in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Use CSV or Excel.")

    return store_dataframe(df, label=label or os.path.basename(file_path))


def get_latest_dataset() -> pd.DataFrame:
    _ensure_dirs()
    meta = _load_meta()
    latest = meta.get("latest_csv")
    if not latest:
        raise FileNotFoundError("No dataset has been uploaded yet.")
    return pd.read_csv(os.path.join(DATA_DIR, latest))


def get_all_datasets() -> pd.DataFrame:
    _ensure_dirs()
    meta = _load_meta()
    frames = []
    for entry in meta.get("uploads", []):
        path = os.path.join(DATA_DIR, entry["path"])
        if os.path.exists(path):
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No datasets available.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates().reset_index(drop=True)


def get_summary() -> Dict[str, Any]:
    meta = _load_meta()
    return {
        "num_uploads": len(meta.get("uploads", [])),
        "latest": meta.get("latest_csv"),
        "uploads": meta.get("uploads", []),
    }
