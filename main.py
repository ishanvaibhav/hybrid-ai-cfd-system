"""
CLI entry point for the CFD prediction system.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from automation import ContinuousFluentScheduler, DEFAULT_RANGES, get_automation_summary, load_condition_plan
from data_handler import get_summary
from pipeline import CFDSystem


def cmd_upload(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    if args.auto_retrain:
        result = system.upload_and_retrain(args.file, label=args.label, pinn_epochs=args.pinn_epochs)
        print(json.dumps(result, indent=2))
    else:
        path = system.upload(args.file, label=args.label)
        print(f"[OK] Uploaded to {path}")
        print(json.dumps(get_summary(), indent=2))


def cmd_train(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    metrics = system.retrain_all(pinn_epochs=args.pinn_epochs)
    print(json.dumps(metrics, indent=2))


def cmd_predict_scalar(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    if args.auto_fallback:
        result = system.full_predict(
            args.velocity,
            args.temperature,
            args.diameter,
            args.valve,
            pinn_epochs=args.pinn_epochs,
            auto_fluent=True,
        )
        print(json.dumps(result, indent=2))
        return

    try:
        result = system.predict_scalar(args.velocity, args.temperature, args.diameter, args.valve)
    except Exception as exc:
        print(f"[WARN] Scalar model unavailable ({exc}), attempting ML training.")
        system.train_ml()
        result = system.predict_scalar(args.velocity, args.temperature, args.diameter, args.valve)
    print(json.dumps(result, indent=2))


def cmd_predict_field(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    try:
        grid = system.predict_field_grid(
            (args.xmin, args.xmax),
            (args.ymin, args.ymax),
            (args.zmin, args.zmax),
            args.velocity,
            args.temperature,
            args.diameter,
            args.valve,
            resolution=args.res,
        )
    except Exception as exc:
        print(f"[WARN] PINN unavailable ({exc}), attempting retraining.")
        pinn_result = system.train_pinn(epochs=args.pinn_epochs)
        if pinn_result.get("skipped"):
            raise RuntimeError(pinn_result["reason"]) from exc
        grid = system.predict_field_grid(
            (args.xmin, args.xmax),
            (args.ymin, args.ymax),
            (args.zmin, args.zmax),
            args.velocity,
            args.temperature,
            args.diameter,
            args.valve,
            resolution=args.res,
        )
    np.savez(args.output, **grid)
    print(f"[OK] Field saved to {args.output}")
    print(f"Grid shape: {grid['p'].shape}")


def cmd_fluent_fallback(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    prediction_context = None
    if args.with_feedback and system.state.get("ml_trained"):
        try:
            prediction_context = system.predict_scalar(args.velocity, args.temperature, args.diameter, args.valve)
        except Exception:
            prediction_context = None

    df = system.run_fluent_fallback(
        args.velocity,
        args.temperature,
        args.diameter,
        args.valve,
        prediction_context=prediction_context,
        feedback_source="manual_cli_fluent",
    )
    print(f"[OK] Fluent fallback complete. Rows added: {len(df)}")
    print(df.head().to_json(orient="records", indent=2))

    if args.retrain:
        metrics = system.retrain_all(pinn_epochs=args.pinn_epochs)
        print(json.dumps({"retrained": True, "metrics": metrics}, indent=2))


def cmd_dashboard(args) -> None:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    command = [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(args.port)]
    if args.headless:
        command.extend(["--server.headless", "true"])
    print(f"[INFO] Launching Streamlit dashboard on port {args.port}...")
    subprocess.run(command, check=False)


def cmd_status(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    print(
        json.dumps(
            {
                "state": system.state,
                "data_summary": get_summary(),
                "feedback_summary": system.get_feedback_summary(),
                "automation_summary": get_automation_summary(),
            },
            indent=2,
        )
    )


def cmd_scheduler(args) -> None:
    system = CFDSystem(force_mock_fluent=args.mock_fluent)
    ranges = {
        "inlet_velocity": args.velocity_range,
        "temperature": args.temperature_range,
        "diameter": args.diameter_range,
        "valve_opening": args.valve_range,
    }
    scheduler = ContinuousFluentScheduler(
        system,
        interval_seconds=args.interval_seconds,
        ranges=ranges,
        retrain_after_each=not args.no_retrain,
        pinn_epochs=args.pinn_epochs,
    )

    conditions = load_condition_plan(args.conditions_file) if args.conditions_file else None
    if args.run_once:
        event = scheduler.run_once(condition=conditions[0] if conditions else None)
        print(json.dumps(event, indent=2))
        return

    scheduler.run_forever(max_runs=args.max_runs, conditions=conditions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CFD Prediction System")
    parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    parser.add_argument("--pinn-epochs", type=int, default=100, help="PINN training epochs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a CSV/Excel dataset")
    upload_parser.add_argument("file", type=str, help="Path to the CSV/Excel file")
    upload_parser.add_argument("--label", type=str, default=None)
    upload_parser.add_argument("--auto-retrain", action="store_true", default=False)
    upload_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    upload_parser.set_defaults(func=cmd_upload)

    train_parser = subparsers.add_parser("train", help="Train ML and PINN models")
    train_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    train_parser.set_defaults(func=cmd_train)

    scalar_parser = subparsers.add_parser("predict_scalar", help="Predict scalar pressure")
    scalar_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    scalar_parser.add_argument("--velocity", type=float, required=True)
    scalar_parser.add_argument("--temperature", type=float, required=True)
    scalar_parser.add_argument("--diameter", type=float, required=True)
    scalar_parser.add_argument("--valve", type=float, required=True)
    scalar_parser.add_argument("--auto-fallback", action="store_true", default=False)
    scalar_parser.set_defaults(func=cmd_predict_scalar)

    field_parser = subparsers.add_parser("predict_field", help="Predict the conditioned field via PINN")
    field_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    field_parser.add_argument("--velocity", type=float, required=True)
    field_parser.add_argument("--temperature", type=float, required=True)
    field_parser.add_argument("--diameter", type=float, required=True)
    field_parser.add_argument("--valve", type=float, required=True)
    field_parser.add_argument("--xmin", type=float, default=0.0)
    field_parser.add_argument("--xmax", type=float, default=1.0)
    field_parser.add_argument("--ymin", type=float, default=0.0)
    field_parser.add_argument("--ymax", type=float, default=0.5)
    field_parser.add_argument("--zmin", type=float, default=0.0)
    field_parser.add_argument("--zmax", type=float, default=0.1)
    field_parser.add_argument("--res", type=int, default=16)
    field_parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "field.npz"),
    )
    field_parser.set_defaults(func=cmd_predict_field)

    fluent_parser = subparsers.add_parser("fluent_fallback", help="Trigger a Fluent simulation")
    fluent_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    fluent_parser.add_argument("--velocity", type=float, required=True)
    fluent_parser.add_argument("--temperature", type=float, required=True)
    fluent_parser.add_argument("--diameter", type=float, required=True)
    fluent_parser.add_argument("--valve", type=float, required=True)
    fluent_parser.add_argument("--retrain", action="store_true", default=False)
    fluent_parser.add_argument("--with-feedback", action="store_true", default=False)
    fluent_parser.set_defaults(func=cmd_fluent_fallback)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    dashboard_parser.add_argument("--headless", action="store_true", default=False)
    dashboard_parser.set_defaults(func=cmd_dashboard)

    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    status_parser.set_defaults(func=cmd_status)

    scheduler_parser = subparsers.add_parser("schedule_fluent", help="Run the continuous Fluent generation loop")
    scheduler_parser.add_argument("--mock-fluent", action="store_true", help="Use mock Fluent instead of real PyFluent")
    scheduler_parser.add_argument("--interval-seconds", type=int, default=3600)
    scheduler_parser.add_argument("--max-runs", type=int, default=None)
    scheduler_parser.add_argument("--run-once", action="store_true", default=False)
    scheduler_parser.add_argument("--no-retrain", action="store_true", default=False)
    scheduler_parser.add_argument("--conditions-file", type=str, default=None)
    scheduler_parser.add_argument("--velocity-range", type=float, nargs=2, default=DEFAULT_RANGES["inlet_velocity"])
    scheduler_parser.add_argument("--temperature-range", type=float, nargs=2, default=DEFAULT_RANGES["temperature"])
    scheduler_parser.add_argument("--diameter-range", type=float, nargs=2, default=DEFAULT_RANGES["diameter"])
    scheduler_parser.add_argument("--valve-range", type=float, nargs=2, default=DEFAULT_RANGES["valve_opening"])
    scheduler_parser.set_defaults(func=cmd_scheduler)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
