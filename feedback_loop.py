"""
Feedback calibration utilities for scalar CFD predictions.

The pipeline records Fluent-validated outcomes and uses recent residuals to
apply a lightweight correction to future ML predictions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

import numpy as np

FEATURE_COLS = ["inlet_velocity", "temperature", "diameter", "valve_opening"]
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback_log.json")


def _ensure_dirs() -> None:
    os.makedirs(FEEDBACK_DIR, exist_ok=True)


def _load_feedback() -> Dict[str, Any]:
    _ensure_dirs()
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
        except JSONDecodeError:
            corrupt_path = f"{FEEDBACK_PATH}.corrupt"
            try:
                os.replace(FEEDBACK_PATH, corrupt_path)
            except OSError:
                pass
    return {"events": []}


def _save_feedback(payload: Dict[str, Any]) -> None:
    _ensure_dirs()
    temp_path = f"{FEEDBACK_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temp_path, FEEDBACK_PATH)


def _vector_from_inputs(input_parameters: Dict[str, float]) -> np.ndarray:
    return np.array([float(input_parameters.get(column, 0.0)) for column in FEATURE_COLS], dtype=float)


def record_feedback(
    input_parameters: Dict[str, float],
    predicted_pressure: float,
    actual_pressure: float,
    source: str = "fluent",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _load_feedback()
    error = float(actual_pressure) - float(predicted_pressure)
    abs_error = abs(error)
    denom = max(abs(float(actual_pressure)), 1.0)
    abs_pct_error = abs_error / denom

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "inputs": {column: float(input_parameters[column]) for column in FEATURE_COLS},
        "predicted_pressure": float(predicted_pressure),
        "actual_pressure": float(actual_pressure),
        "error": float(error),
        "abs_error": float(abs_error),
        "abs_pct_error": float(abs_pct_error),
        "metadata": metadata or {},
    }
    payload["events"].append(event)
    _save_feedback(payload)
    return event


def get_feedback_summary(window: int = 20) -> Dict[str, Any]:
    payload = _load_feedback()
    events = payload.get("events", [])
    recent = events[-window:]

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    abs_errors = [float(event["abs_error"]) for event in recent]
    pct_errors = [float(event["abs_pct_error"]) for event in recent]
    signed_errors = [float(event["error"]) for event in recent]

    sources: Dict[str, int] = {}
    for event in events:
        sources[event.get("source", "unknown")] = sources.get(event.get("source", "unknown"), 0) + 1

    return {
        "total_events": len(events),
        "recent_window": window,
        "recent_events": len(recent),
        "recent_mae": _mean(abs_errors),
        "recent_mape": _mean(pct_errors),
        "recent_mean_error": _mean(signed_errors),
        "latest_event": recent[-1] if recent else None,
        "sources": sources,
    }


def estimate_feedback_adjustment(
    input_parameters: Dict[str, float],
    max_neighbors: int = 5,
) -> Dict[str, Any]:
    payload = _load_feedback()
    events = payload.get("events", [])
    if not events:
        return {
            "adjustment": 0.0,
            "samples_used": 0,
            "method": "none",
            "nearest_distance": None,
        }

    vectors = []
    errors = []
    for event in events:
        inputs = event.get("inputs", {})
        if all(column in inputs for column in FEATURE_COLS):
            vectors.append(_vector_from_inputs(inputs))
            errors.append(float(event.get("error", 0.0)))

    if not vectors:
        return {
            "adjustment": 0.0,
            "samples_used": 0,
            "method": "none",
            "nearest_distance": None,
        }

    matrix = np.vstack(vectors)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0.0] = 1.0

    query = _vector_from_inputs(input_parameters)
    distances = np.linalg.norm((matrix - center) / scale - (query - center) / scale, axis=1)
    order = np.argsort(distances)[: max(1, min(max_neighbors, len(distances)))]
    chosen_distances = distances[order]
    chosen_errors = np.array(errors, dtype=float)[order]
    weights = 1.0 / (chosen_distances + 1.0e-6)
    adjustment = float(np.dot(weights, chosen_errors) / weights.sum())

    return {
        "adjustment": adjustment,
        "samples_used": int(len(order)),
        "method": "knn_residual",
        "nearest_distance": float(chosen_distances[0]) if len(chosen_distances) else None,
    }
