"""
Background automation helpers for continuous Fluent data generation.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from feedback_loop import get_feedback_summary

AUTOMATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automation")
AUTOMATION_LOG_PATH = os.path.join(AUTOMATION_DIR, "automation_log.json")
DEFAULT_RANGES = {
    "inlet_velocity": [0.5, 10.0],
    "temperature": [20.0, 100.0],
    "diameter": [0.05, 0.3],
    "valve_opening": [0.2, 1.0],
}


def _ensure_dirs() -> None:
    os.makedirs(AUTOMATION_DIR, exist_ok=True)


def _load_log() -> Dict[str, Any]:
    _ensure_dirs()
    if os.path.exists(AUTOMATION_LOG_PATH):
        with open(AUTOMATION_LOG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"events": []}


def _save_log(payload: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(AUTOMATION_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def get_automation_summary(window: int = 10) -> Dict[str, Any]:
    payload = _load_log()
    events = payload.get("events", [])
    recent = events[-window:]
    return {
        "total_runs": len(events),
        "recent_runs": len(recent),
        "latest_event": recent[-1] if recent else None,
    }


def sample_conditions(
    ranges: Optional[Dict[str, List[float]]] = None,
    count: int = 1,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    bounds = ranges or DEFAULT_RANGES
    conditions: List[Dict[str, float]] = []
    for _ in range(count):
        conditions.append(
            {
                name: float(rng.uniform(limit[0], limit[1]))
                for name, limit in bounds.items()
            }
        )
    return conditions


def load_condition_plan(path: str) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "conditions" in payload:
        return payload["conditions"]
    raise ValueError("Conditions file must be a JSON list or a JSON object with a 'conditions' key.")


class ContinuousFluentScheduler:
    def __init__(
        self,
        system,
        interval_seconds: int = 3600,
        ranges: Optional[Dict[str, List[float]]] = None,
        retrain_after_each: bool = True,
        pinn_epochs: int = 100,
    ) -> None:
        self.system = system
        self.interval_seconds = interval_seconds
        self.ranges = ranges or DEFAULT_RANGES
        self.retrain_after_each = retrain_after_each
        self.pinn_epochs = pinn_epochs

    def run_once(self, condition: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        chosen = condition or sample_conditions(self.ranges, count=1)[0]

        prediction_context = None
        if self.system.state.get("ml_trained"):
            try:
                prediction_context = self.system.predict_scalar(
                    chosen["inlet_velocity"],
                    chosen["temperature"],
                    chosen["diameter"],
                    chosen["valve_opening"],
                )
            except Exception:
                prediction_context = None

        df = self.system.run_fluent_fallback(
            chosen["inlet_velocity"],
            chosen["temperature"],
            chosen["diameter"],
            chosen["valve_opening"],
            prediction_context=prediction_context,
            feedback_source="scheduled_fluent_loop",
        )

        retrain_metrics = None
        if self.retrain_after_each:
            retrain_metrics = self.system.retrain_all(pinn_epochs=self.pinn_epochs)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "condition": chosen,
            "rows": int(len(df)),
            "retrained": bool(self.retrain_after_each),
            "feedback_summary": get_feedback_summary(),
            "retrain_metrics": retrain_metrics,
        }
        payload = _load_log()
        payload["events"].append(event)
        _save_log(payload)
        return event

    def run_forever(
        self,
        max_runs: Optional[int] = None,
        conditions: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        completed = 0
        cursor = 0
        while max_runs is None or completed < max_runs:
            condition = None
            if conditions:
                condition = conditions[cursor % len(conditions)]
                cursor += 1
            self.run_once(condition=condition)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break
            time.sleep(self.interval_seconds)
