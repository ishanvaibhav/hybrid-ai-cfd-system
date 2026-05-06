"""
PyFluent integration layer for automated CFD data generation.

The wrapper attempts a real Fluent solve first and falls back to a deterministic
mock implementation when PyFluent or the configured case is unavailable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ansys.fluent.core as pyfluent

    HAS_PYFLUENT = True
except Exception:
    HAS_PYFLUENT = False

FLUENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fluent_cases")
META_PATH = os.path.join(FLUENT_DIR, "fluent_meta.json")
CONFIG_PATH = os.path.join(FLUENT_DIR, "fluent_config.json")
MOCK_PRESSURE_MULTIPLIER = 1200.0
EXPORT_COLUMNS = [
    "x-coordinate",
    "y-coordinate",
    "z-coordinate",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "pressure",
]
RENAMED_COLUMNS = {
    "x-coordinate": "x",
    "y-coordinate": "y",
    "z-coordinate": "z",
    "x-velocity": "u",
    "y-velocity": "v",
    "z-velocity": "w",
    "pressure": "p",
}


def _ensure_dirs() -> None:
    os.makedirs(FLUENT_DIR, exist_ok=True)


def _default_config() -> Dict[str, Any]:
    return {
        "case_file": os.path.join(FLUENT_DIR, "base_case.cas.h5"),
        "data_file": None,
        "inlet_name": "inlet",
        "outlet_name": "outlet",
        "pressure_outlet_gauge_pressure": 0.0,
        "iterations": 200,
        "export_surfaces": ["inlet", "outlet"],
        "export_location": "node",
        "export_delimiter": ",",
        "archive_case_data": True,
        "case_parameter_paths": {
            "diameter": [],
            "valve_opening": [],
        },
    }


def _load_config() -> Dict[str, Any]:
    _ensure_dirs()
    config = _default_config()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config.update(json.load(file))
    return config


def _load_meta() -> Dict[str, Any]:
    _ensure_dirs()
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"runs": []}


def _save_meta(meta: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(META_PATH, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def _mock_results() -> pd.DataFrame:
    n_points = 200
    x = np.linspace(0.0, 1.0, n_points)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.zeros(n_points),
            "z": np.zeros(n_points),
            "u": np.ones(n_points) * 0.5,
            "v": np.zeros(n_points),
            "w": np.zeros(n_points),
            "p": np.sin(x * np.pi) * 1000.0 + 500.0,
        }
    )


def _normalize_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    rename_map = {source.lower(): target for source, target in RENAMED_COLUMNS.items()}
    normalized = normalized.rename(columns=rename_map)
    return normalized


class FluentSession:
    def __init__(self, mock: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or _load_config()
        self.mock = mock or not HAS_PYFLUENT
        self.session = None

        if not self.mock:
            try:
                self.session = pyfluent.launch_fluent(
                    mode="solver",
                    precision="double",
                    processor_count=4,
                    show_gui=False,
                )
            except Exception as exc:
                print(f"[Fluent] Could not launch Fluent: {exc}. Falling back to mock mode.")
                self.mock = True

    def _set_case_parameter(self, parameter_name: str, value: float) -> None:
        path_tokens = self.config.get("case_parameter_paths", {}).get(parameter_name) or []
        if not path_tokens or self.session is None:
            return

        current = self.session
        try:
            for token in path_tokens[:-1]:
                current = getattr(current, token)
            leaf = getattr(current, path_tokens[-1])
            if hasattr(leaf, "set_state"):
                leaf.set_state(value)
            else:
                setattr(current, path_tokens[-1], value)
        except Exception as exc:
            print(f"[Fluent] Could not set case parameter '{parameter_name}': {exc}")

    def _read_case_data(self) -> None:
        if self.session is None:
            return

        case_file = self.config.get("case_file")
        data_file = self.config.get("data_file")
        if data_file:
            self.session.settings.file.read_case(file_name=case_file)
            self.session.settings.file.read_data(file_name=data_file)
        else:
            self.session.settings.file.read_case(file_name=case_file)

    def _configure_boundary_conditions(
        self,
        inlet_velocity: float,
        temperature: float,
    ) -> None:
        if self.session is None:
            return

        inlet_name = self.config.get("inlet_name", "inlet")
        outlet_name = self.config.get("outlet_name", "outlet")
        outlet_pressure = float(self.config.get("pressure_outlet_gauge_pressure", 0.0))

        try:
            velocity_inlet = pyfluent.VelocityInlet(settings_source=self.session, name=inlet_name)
            velocity_inlet.momentum = {"velocity_magnitude": {"value": float(inlet_velocity)}}
            velocity_inlet.thermal = {"temperature": {"value": float(temperature)}}
        except Exception as exc:
            print(f"[Fluent] Velocity inlet update failed: {exc}")

        try:
            pressure_outlet = pyfluent.PressureOutlet(settings_source=self.session, name=outlet_name)
            pressure_outlet.momentum = {"gauge_pressure": {"value": outlet_pressure}}
        except Exception as exc:
            print(f"[Fluent] Pressure outlet update failed: {exc}")

    def setup_case(
        self,
        inlet_velocity: float,
        temperature: float,
        diameter: float,
        valve_opening: float,
    ) -> Dict[str, Any]:
        meta = {
            "inlet_velocity": inlet_velocity,
            "temperature": temperature,
            "diameter": diameter,
            "valve_opening": valve_opening,
        }

        if self.mock:
            meta["mode"] = "mock"
            return meta

        case_file = self.config.get("case_file")
        if not case_file or not os.path.exists(case_file):
            self.mock = True
            meta["mode"] = "mock_no_case_file"
            return meta

        try:
            self._read_case_data()
            self._configure_boundary_conditions(inlet_velocity, temperature)
            self._set_case_parameter("diameter", diameter)
            self._set_case_parameter("valve_opening", valve_opening)
            meta["mode"] = "real"
        except Exception as exc:
            print(f"[Fluent] Case setup error: {exc}. Switching to mock mode.")
            self.mock = True
            meta["mode"] = "mock_case_error"
        return meta

    def _initialize_solution(self) -> None:
        if self.session is None:
            return

        try:
            self.session.solution.initialization.hybrid_initialize()
            return
        except Exception:
            pass

        try:
            self.session.tui.solve.initialize.hyb_initialization()
        except Exception as exc:
            print(f"[Fluent] Initialization warning: {exc}")

    def solve(self, iterations: Optional[int] = None) -> Dict[str, Any]:
        iterations = int(iterations or self.config.get("iterations", 200))
        if self.mock:
            return {"iterations": iterations, "mode": "mock"}

        try:
            self._initialize_solution()
            try:
                self.session.solution.run_calculation.calculate(iter_count=iterations)
            except Exception:
                self.session.tui.solve.iterate(str(iterations))
            return {"iterations": iterations, "mode": "real"}
        except Exception as exc:
            print(f"[Fluent] Solve error: {exc}. Returning mock results instead.")
            self.mock = True
            return {"iterations": iterations, "mode": "mock_after_solve_error"}

    def _archive_solution(self, stem: str) -> Optional[str]:
        if self.session is None or self.mock or not self.config.get("archive_case_data", True):
            return None

        archive_path = os.path.join(FLUENT_DIR, f"{stem}.casdat.h5")
        try:
            self.session.settings.file.write_case_data(file_name=archive_path)
            return archive_path
        except Exception as exc:
            print(f"[Fluent] Case-data archive warning: {exc}")
            return None

    def export_results(self, export_stem: str = "fluent_export") -> pd.DataFrame:
        if self.mock:
            return _mock_results()

        export_path = os.path.join(FLUENT_DIR, f"{export_stem}.csv")
        try:
            self.session.settings.file.export.ascii(
                file_name=export_path,
                surface_name_list=self.config.get("export_surfaces", []),
                delimiter=self.config.get("export_delimiter", ","),
                quantities=EXPORT_COLUMNS,
                location=self.config.get("export_location", "node"),
            )
            self._archive_solution(export_stem)
            df = pd.read_csv(export_path)
            return _normalize_export_frame(df)
        except Exception as exc:
            print(f"[Fluent] Settings export failed: {exc}. Trying TUI ASCII export.")

        try:
            self.session.tui.file.export.ascii(
                export_path,
                self.config.get("export_surfaces", []),
                self.config.get("export_delimiter", ","),
                EXPORT_COLUMNS,
                self.config.get("export_location", "node"),
            )
            self._archive_solution(export_stem)
            df = pd.read_csv(export_path)
            return _normalize_export_frame(df)
        except Exception as exc:
            print(f"[Fluent] Export error: {exc}. Returning mock results instead.")
            self.mock = True
            return _mock_results()

    def close(self) -> None:
        if self.session is not None:
            try:
                self.session.exit()
            except Exception:
                pass

    def __enter__(self) -> "FluentSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def run_simulation(
    inlet_velocity: float,
    temperature: float,
    diameter: float,
    valve_opening: float,
    force_mock: bool = False,
) -> pd.DataFrame:
    _ensure_dirs()
    config = _load_config()
    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    with FluentSession(mock=force_mock, config=config) as fluent:
        setup_meta = fluent.setup_case(inlet_velocity, temperature, diameter, valve_opening)
        solve_meta = fluent.solve(iterations=config.get("iterations", 200))
        df = fluent.export_results(export_stem=f"fluent_export_{run_stamp}")

    df = df.copy()
    df["inlet_velocity"] = inlet_velocity
    df["temperature"] = temperature
    df["diameter"] = diameter
    df["valve_opening"] = valve_opening

    if "max_pressure" not in df.columns:
        if "p" in df.columns:
            df["max_pressure"] = float(df["p"].max())
        else:
            df["max_pressure"] = inlet_velocity * MOCK_PRESSURE_MULTIPLIER / max(valve_opening, 0.1)

    meta = _load_meta()
    meta["runs"].append(
        {
            "timestamp": run_stamp,
            "inlet_velocity": inlet_velocity,
            "temperature": temperature,
            "diameter": diameter,
            "valve_opening": valve_opening,
            "setup_mode": setup_meta.get("mode", "unknown"),
            "solve_mode": solve_meta.get("mode", "unknown"),
            "rows": int(len(df)),
        }
    )
    _save_meta(meta)
    return df


def generate_training_data_from_fluent(
    conditions: List[Dict[str, float]],
    force_mock: bool = False,
) -> pd.DataFrame:
    frames = []
    for condition in conditions:
        frames.append(
            run_simulation(
                condition["inlet_velocity"],
                condition["temperature"],
                condition["diameter"],
                condition["valve_opening"],
                force_mock=force_mock,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False)
