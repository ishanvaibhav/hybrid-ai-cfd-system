"""
Machine-learning module for scalar CFD predictions.

The regressor predicts max pressure from operating conditions and basic
geometry, then maps the predicted pressure into HIGH / NORMAL / LOW bands.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ML_MODEL_PATH = os.path.join(MODEL_DIR, "ml_regressor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "ml_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "ml_meta.json")

DEFAULT_THRESHOLD_HIGH = 5000.0
DEFAULT_THRESHOLD_LOW = 1000.0

FEATURE_COLS = ["inlet_velocity", "temperature", "diameter", "valve_opening"]
TARGET_COL = "max_pressure"


def _ensure_dirs() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def _load_meta() -> Dict[str, Any]:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def _save_meta(meta: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(META_PATH, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


class MLPredictor:
    def __init__(
        self,
        threshold_high: float = DEFAULT_THRESHOLD_HIGH,
        threshold_low: float = DEFAULT_THRESHOLD_LOW,
    ) -> None:
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[xgb.XGBRegressor] = None
        self.is_trained = False

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        _ensure_dirs()

        missing = [column for column in FEATURE_COLS + [TARGET_COL] if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for ML training: {missing}")
        if len(df) < 5:
            raise ValueError("At least 5 rows are required for ML training.")

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        y_pred_class = np.array([self._classify_scalar(value) for value in y_pred])
        y_true_class = np.array([self._classify_scalar(value) for value in y_test])
        accuracy = float(accuracy_score(y_true_class, y_pred_class))

        meta = {
            "rmse": rmse,
            "r2": r2,
            "classification_accuracy": accuracy,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
        }

        self.save(extra_meta=meta)
        return meta

    def predict_pressure(self, input_parameters: Dict[str, float]) -> Dict[str, Any]:
        if not self.is_trained or self.model is None or self.scaler is None:
            raise RuntimeError("Model is not trained yet. Call train() or load() first.")

        vector = np.array(
            [[float(input_parameters.get(column, 0.0)) for column in FEATURE_COLS]],
            dtype=np.float32,
        )
        vector_scaled = self.scaler.transform(vector)
        prediction = float(self.model.predict(vector_scaled)[0])
        status = self._classify_scalar(prediction)
        return {
            "predicted_pressure": prediction,
            "status": status,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained or self.model is None or self.scaler is None:
            raise RuntimeError("Model is not trained yet. Call train() or load() first.")

        missing = [column for column in FEATURE_COLS if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for batch prediction: {missing}")

        X = df[FEATURE_COLS].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        output = df.copy()
        output["predicted_pressure"] = predictions
        output["status"] = [self._classify_scalar(value) for value in predictions]
        return output

    def _classify_scalar(self, pressure: float) -> str:
        if pressure > self.threshold_high:
            return "HIGH"
        if pressure < self.threshold_low:
            return "LOW"
        return "NORMAL"

    def save(self, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        _ensure_dirs()
        if self.model is None or self.scaler is None:
            raise RuntimeError("No trained ML model is available to save.")

        joblib.dump(self.model, ML_MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        meta = _load_meta()
        meta.update(
            {
                "threshold_high": self.threshold_high,
                "threshold_low": self.threshold_low,
            }
        )
        if extra_meta:
            meta.update(extra_meta)
        _save_meta(meta)

    def load(self) -> None:
        if not os.path.exists(ML_MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("No saved ML model found. Train first.")

        self.model = joblib.load(ML_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        meta = _load_meta()
        self.threshold_high = meta.get("threshold_high", DEFAULT_THRESHOLD_HIGH)
        self.threshold_low = meta.get("threshold_low", DEFAULT_THRESHOLD_LOW)
        self.is_trained = True

    def feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained.")
        importance = self.model.feature_importances_
        return {name: float(value) for name, value in zip(FEATURE_COLS, importance)}


def get_trained_predictor() -> MLPredictor:
    predictor = MLPredictor()
    try:
        predictor.load()
    except FileNotFoundError:
        pass
    return predictor
