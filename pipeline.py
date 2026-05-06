"""
Unified pipeline orchestrator for the CFD prediction system.

Workflow:
    Upload data -> train models -> predict -> trigger Fluent fallback -> feed back
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from data_handler import get_all_datasets, get_summary, store_dataframe, upload_data
from feedback_loop import estimate_feedback_adjustment, get_feedback_summary, record_feedback
from fluent_wrapper import run_simulation
from ml_module import MLPredictor, get_trained_predictor
from pinn_module import CONDITION_COLS, PINN_INPUT_COLS, PINNTrainer, get_trained_pinn

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline")
STATE_PATH = os.path.join(PIPELINE_DIR, "state.json")
UNCERTAINTY_THRESHOLD_HIGH = 0.20


def _ensure_dirs() -> None:
    os.makedirs(PIPELINE_DIR, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    default_state = {
        "ml_trained": False,
        "pinn_trained": False,
        "last_upload": None,
        "fluent_runs": 0,
        "feedback_events": 0,
        "last_feedback": None,
    }
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as file:
            default_state.update(json.load(file))
            return default_state
    return default_state


def _save_state(state: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


class CFDSystem:
    def __init__(
        self,
        ml_threshold_high: float = 5000.0,
        ml_threshold_low: float = 1000.0,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD_HIGH,
        force_mock_fluent: bool = False,
    ) -> None:
        self.ml_predictor = MLPredictor(
            threshold_high=ml_threshold_high,
            threshold_low=ml_threshold_low,
        )
        self.pinn_trainer = PINNTrainer()
        self.uncertainty_threshold = uncertainty_threshold
        self.force_mock_fluent = force_mock_fluent
        self.state = _load_state()
        self._hydrate_models()

    def _hydrate_models(self) -> None:
        trained_ml = get_trained_predictor()
        if trained_ml.is_trained:
            self.ml_predictor = trained_ml
            self.state["ml_trained"] = True

        trained_pinn = get_trained_pinn()
        if trained_pinn.is_trained:
            self.pinn_trainer = trained_pinn
            self.state["pinn_trained"] = True

        _save_state(self.state)

    def _ensure_ml_loaded(self) -> None:
        if self.ml_predictor.is_trained:
            return
        self.ml_predictor.load()
        self.state["ml_trained"] = True
        _save_state(self.state)

    def _ensure_pinn_loaded(self) -> None:
        if self.pinn_trainer.is_trained:
            return
        self.pinn_trainer.load()
        self.state["pinn_trained"] = True
        _save_state(self.state)

    def _build_conditions(
        self,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
    ) -> Dict[str, float]:
        return {
            "inlet_velocity": float(inlet_velocity),
            "temperature": float(temperature),
            "diameter": float(diameter),
            "valve_opening": float(valve_opening),
        }

    def _can_train_pinn(self, df: pd.DataFrame) -> bool:
        return all(column in df.columns for column in PINN_INPUT_COLS)

    def upload(self, file_path: str, label: Optional[str] = None) -> str:
        path = upload_data(file_path, label=label)
        self.state["last_upload"] = os.path.basename(path)
        _save_state(self.state)
        return path

    def upload_and_retrain(
        self,
        file_path: str,
        label: Optional[str] = None,
        pinn_epochs: int = 100,
    ) -> Dict[str, Any]:
        path = self.upload(file_path, label=label)
        outcome: Dict[str, Any] = {
            "path": path,
            "retrained": False,
            "metrics": None,
            "data_summary": get_summary(),
        }
        try:
            outcome["metrics"] = self.retrain_all(pinn_epochs=pinn_epochs)
            outcome["retrained"] = True
        except Exception as exc:
            outcome["retrain_error"] = str(exc)
        return outcome

    def get_data_summary(self) -> Dict[str, Any]:
        return get_summary()

    def get_feedback_summary(self) -> Dict[str, Any]:
        return get_feedback_summary()

    def train_ml(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if df is None:
            df = get_all_datasets()
        metrics = self.ml_predictor.train(df)
        self.state["ml_trained"] = True
        _save_state(self.state)
        return metrics

    def train_pinn(self, df: Optional[pd.DataFrame] = None, epochs: int = 5000) -> Dict[str, Any]:
        if df is None:
            df = get_all_datasets()

        if not self._can_train_pinn(df):
            self.state["pinn_trained"] = False
            _save_state(self.state)
            return {
                "skipped": True,
                "reason": f"PINN training requires conditioned columns: {PINN_INPUT_COLS}",
            }

        metrics = self.pinn_trainer.train(df, epochs=epochs)
        self.state["pinn_trained"] = True
        _save_state(self.state)
        return metrics

    def retrain_all(
        self,
        df: Optional[pd.DataFrame] = None,
        pinn_epochs: int = 5000,
    ) -> Dict[str, Any]:
        if df is None:
            df = get_all_datasets()
        ml_metrics = self.train_ml(df)
        pinn_metrics = self.train_pinn(df, epochs=pinn_epochs)
        return {"ml": ml_metrics, "pinn": pinn_metrics}

    def predict_scalar(
        self,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
    ) -> Dict[str, Any]:
        self._ensure_ml_loaded()
        conditions = self._build_conditions(inlet_velocity, temperature, diameter, valve_opening)

        result = self.ml_predictor.predict_pressure(conditions)
        raw_prediction = float(result["predicted_pressure"])

        feedback_adjustment = estimate_feedback_adjustment(conditions)
        raw_adjustment = float(feedback_adjustment["adjustment"])
        feedback_gain = min(max(feedback_adjustment["samples_used"], 0), 5) / 5.0
        feedback_gain = max(feedback_gain, 0.2) * 0.35 if feedback_adjustment["samples_used"] > 0 else 0.0
        max_adjustment = max(abs(raw_prediction) * 0.2, 200.0)
        applied_adjustment = float(np.clip(raw_adjustment * feedback_gain, -max_adjustment, max_adjustment))
        corrected_prediction = raw_prediction + applied_adjustment
        result["predicted_pressure_raw"] = raw_prediction
        result["predicted_pressure"] = corrected_prediction
        result["feedback_adjustment_raw"] = raw_adjustment
        result["feedback_adjustment"] = applied_adjustment
        result["feedback_samples_used"] = int(feedback_adjustment["samples_used"])
        result["feedback_method"] = feedback_adjustment["method"]
        result["status"] = self.ml_predictor._classify_scalar(corrected_prediction)

        try:
            df = get_all_datasets()
            feature_cols = CONDITION_COLS
            means = df[feature_cols].mean()
            stds = df[feature_cols].std()
            vector = np.array([conditions[column] for column in feature_cols], dtype=float)
            scale = stds.values.astype(float)
            scale[scale == 0.0] = 1.0
            z_score = np.linalg.norm((vector - means.values.astype(float)) / scale)
            base_uncertainty = min(z_score / 3.0, 1.0)
        except Exception:
            base_uncertainty = 0.0

        feedback_summary = get_feedback_summary()
        feedback_uncertainty = min(float(feedback_summary.get("recent_mape", 0.0)), 1.0)
        calibrated_uncertainty = min(max(base_uncertainty, feedback_uncertainty), 1.0)
        result["uncertainty_raw"] = round(float(base_uncertainty), 4)
        result["uncertainty"] = round(float(calibrated_uncertainty), 4)
        result["feedback_recent_mape"] = round(float(feedback_summary.get("recent_mape", 0.0)), 4)
        result["trigger_fluent"] = bool(calibrated_uncertainty > self.uncertainty_threshold)
        return result

    def predict_field(
        self,
        coordinates: np.ndarray,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
    ) -> Dict[str, np.ndarray]:
        self._ensure_pinn_loaded()
        conditions = self._build_conditions(inlet_velocity, temperature, diameter, valve_opening)
        return self.pinn_trainer.predict_field(coordinates, conditions)

    def predict_field_grid(
        self,
        x_range,
        y_range,
        z_range,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
        resolution: int = 32,
    ) -> Dict[str, np.ndarray]:
        self._ensure_pinn_loaded()
        conditions = self._build_conditions(inlet_velocity, temperature, diameter, valve_opening)
        return self.pinn_trainer.predict_grid(x_range, y_range, z_range, conditions, resolution)

    def run_fluent_fallback(
        self,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
        prediction_context: Optional[Dict[str, Any]] = None,
        feedback_source: str = "fluent_fallback",
    ) -> pd.DataFrame:
        df = run_simulation(
            inlet_velocity,
            temperature,
            diameter,
            valve_opening,
            force_mock=self.force_mock_fluent,
        )
        label = f"fluent_fallback_{self.state['fluent_runs']}"
        store_dataframe(df, label=label)
        self.state["fluent_runs"] += 1
        self.state["last_upload"] = label

        if prediction_context is not None and "predicted_pressure" in prediction_context:
            conditions = self._build_conditions(inlet_velocity, temperature, diameter, valve_opening)
            actual_pressure = float(df["max_pressure"].max()) if "max_pressure" in df.columns else float(df["p"].max())
            feedback_event = record_feedback(
                input_parameters=conditions,
                predicted_pressure=float(prediction_context["predicted_pressure"]),
                actual_pressure=actual_pressure,
                source=feedback_source,
                metadata={
                    "predicted_pressure_raw": float(prediction_context.get("predicted_pressure_raw", prediction_context["predicted_pressure"])),
                    "uncertainty": float(prediction_context.get("uncertainty", 0.0)),
                },
            )
            self.state["feedback_events"] = int(self.state.get("feedback_events", 0)) + 1
            self.state["last_feedback"] = feedback_event

        _save_state(self.state)
        return df

    def full_predict(
        self,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
        coordinates: Optional[np.ndarray] = None,
        pinn_epochs: int = 100,
        auto_fluent: bool = True,
    ) -> Dict[str, Any]:
        initial_scalar = self.predict_scalar(inlet_velocity, temperature, diameter, valve_opening)
        retrain_metrics = None
        fluent_triggered = bool(initial_scalar["trigger_fluent"] and auto_fluent)
        scalar = initial_scalar

        if fluent_triggered:
            print("[Pipeline] High uncertainty detected. Triggering Fluent simulation.")
            self.run_fluent_fallback(
                inlet_velocity,
                temperature,
                diameter,
                valve_opening,
                prediction_context=initial_scalar,
                feedback_source="automatic_fluent_loop",
            )
            retrain_metrics = self.retrain_all(pinn_epochs=pinn_epochs)
            scalar = self.predict_scalar(inlet_velocity, temperature, diameter, valve_opening)

        result: Dict[str, Any] = {
            "scalar_initial": initial_scalar,
            "scalar": scalar,
            "field": None,
            "fluent_triggered": fluent_triggered,
            "retrain_metrics": retrain_metrics,
            "feedback_summary": get_feedback_summary(),
        }
        if coordinates is not None and self.state.get("pinn_trained"):
            result["field"] = self.predict_field(
                coordinates,
                inlet_velocity,
                temperature,
                diameter,
                valve_opening,
            )
        return result

    def save_models(self) -> None:
        if self.ml_predictor.is_trained:
            self.ml_predictor.save()
        if self.pinn_trainer.is_trained:
            self.pinn_trainer.save()
        print("[Pipeline] Models saved.")

    def load_models(self) -> None:
        ml_loaded = False
        pinn_loaded = False

        try:
            self.ml_predictor.load()
            ml_loaded = True
        except FileNotFoundError:
            pass

        try:
            self.pinn_trainer.load()
            pinn_loaded = True
        except FileNotFoundError:
            pass

        self.state["ml_trained"] = ml_loaded
        self.state["pinn_trained"] = pinn_loaded
        _save_state(self.state)

        if not ml_loaded and not pinn_loaded:
            raise FileNotFoundError("No saved ML or PINN models were found.")
