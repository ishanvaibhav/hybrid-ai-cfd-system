"""
Streamlit dashboard for the CFD prediction system.
"""

from __future__ import annotations

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automation import get_automation_summary
from data_handler import get_latest_dataset, get_summary
from pipeline import CFDSystem

st.set_page_config(page_title="CFD Prediction System", layout="wide")


def get_system() -> CFDSystem:
    if "cfd_system" not in st.session_state:
        st.session_state["cfd_system"] = CFDSystem(force_mock_fluent=True)
    return st.session_state["cfd_system"]


def write_uploaded_temp_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name


def generate_demo_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n_rows = 500
    inlet_velocity = np.random.uniform(0.5, 5.0, n_rows)
    temperature = np.random.uniform(20.0, 80.0, n_rows)
    diameter = np.random.uniform(0.05, 0.3, n_rows)
    valve_opening = np.random.uniform(0.2, 1.0, n_rows)
    max_pressure = (
        1000.0
        + 800.0 * inlet_velocity
        + 5.0 * temperature
        - 2000.0 * valve_opening
        + 500.0 * diameter
        + np.random.normal(0.0, 200.0, n_rows)
    )
    x = np.random.uniform(0.0, 1.0, n_rows)
    y = np.random.uniform(0.0, 0.5, n_rows)
    z = np.random.uniform(0.0, 0.1, n_rows)
    u = inlet_velocity * (1.0 - 0.2 * valve_opening)
    return pd.DataFrame(
        {
            "inlet_velocity": inlet_velocity,
            "temperature": temperature,
            "diameter": diameter,
            "valve_opening": valve_opening,
            "max_pressure": max_pressure,
            "x": x,
            "y": y,
            "z": z,
            "u": u,
            "v": np.zeros(n_rows),
            "w": np.zeros(n_rows),
            "p": max_pressure * np.sin(x * np.pi) * 0.8 + np.random.normal(0.0, 50.0, n_rows),
        }
    )


st.sidebar.title("CFD Prediction System")
page = st.sidebar.radio(
    "Navigation",
    [
        "Upload Data",
        "Train Models",
        "Scalar Prediction",
        "Field Prediction",
        "Fluent Fallback",
        "System Status",
    ],
)
st.sidebar.caption("Tech stack: PyTorch, XGBoost, PyFluent, Streamlit")

system = get_system()

if page == "Upload Data":
    st.header("1. Upload Dataset")
    st.markdown("Upload a CSV or Excel file with columns:")
    st.code(
        "inlet_velocity, temperature, diameter, valve_opening, max_pressure, [x, y, z, u, v, w, p]",
        language="text",
    )
    auto_retrain = st.checkbox("Auto-retrain after upload", value=True)
    upload_pinn_epochs = st.slider("PINN retrain epochs after upload", 10, 2000, 100, 10)

    uploaded_file = st.file_uploader("Choose CSV or Excel", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        temp_path = write_uploaded_temp_file(uploaded_file)
        try:
            if auto_retrain:
                upload_result = system.upload_and_retrain(
                    temp_path,
                    label=uploaded_file.name,
                    pinn_epochs=upload_pinn_epochs,
                )
                st.success("File uploaded and automatic retraining completed.")
                st.write(f"Stored at: `{upload_result['path']}`")
                if upload_result.get("metrics") is not None:
                    st.subheader("Retraining Metrics")
                    st.json(upload_result["metrics"])
                if upload_result.get("retrain_error"):
                    st.warning(upload_result["retrain_error"])
            else:
                output_path = system.upload(temp_path, label=uploaded_file.name)
                st.success("File uploaded and validated.")
                st.write(f"Stored at: `{output_path}`")

            dataset = get_latest_dataset()
            st.subheader("Preview")
            st.dataframe(dataset.head(10), use_container_width=True)
            st.subheader("Column Summary")
            st.json(
                {
                    "rows": int(len(dataset)),
                    "columns": list(dataset.columns),
                    "missing": dataset.isna().sum().to_dict(),
                }
            )
        except Exception as exc:
            st.error(f"Upload error: {exc}")

    if st.button("Generate Synthetic Demo Data"):
        demo_df = generate_demo_dataset()
        demo_path = os.path.join(tempfile.gettempdir(), "demo_cfd_data.csv")
        demo_df.to_csv(demo_path, index=False)
        result = system.upload_and_retrain(demo_path, label="synthetic_demo", pinn_epochs=upload_pinn_epochs)
        st.success("Synthetic demo data generated and ingested.")
        if result.get("metrics") is not None:
            st.json(result["metrics"])
        st.dataframe(demo_df.head(10), use_container_width=True)

elif page == "Train Models":
    st.header("2. Train Models")
    epochs = st.slider("PINN training epochs", 10, 5000, 500, 10)
    if st.button("Train ML + PINN"):
        try:
            with st.spinner("Training ML regressor..."):
                ml_metrics = system.train_ml()
            st.success("ML training complete.")
            st.json(ml_metrics)

            with st.spinner("Training PINN..."):
                pinn_metrics = system.train_pinn(epochs=epochs)

            if pinn_metrics.get("skipped"):
                st.warning(pinn_metrics["reason"])
            else:
                st.success("PINN training complete.")
                st.json(pinn_metrics)

            st.subheader("ML Feature Importance")
            importance = system.ml_predictor.feature_importance()
            fig, ax = plt.subplots()
            ax.barh(list(importance.keys()), list(importance.values()))
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Training error: {exc}")

elif page == "Scalar Prediction":
    st.header("3. Scalar Prediction (Closed Loop)")
    col1, col2 = st.columns(2)
    with col1:
        inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.1, 10.0, 2.0, 0.1)
        temperature = st.slider("Temperature (C)", 10.0, 100.0, 25.0, 1.0)
    with col2:
        diameter = st.slider("Diameter (m)", 0.01, 0.5, 0.1, 0.01)
        valve_opening = st.slider("Valve Opening (fraction)", 0.0, 1.0, 0.5, 0.05)

    auto_fluent = st.checkbox("Auto-run Fluent and retrain when uncertainty is high", value=True)
    fallback_pinn_epochs = st.slider("PINN epochs for automatic retraining", 10, 2000, 100, 10)

    if st.button("Predict Pressure"):
        try:
            result = system.full_predict(
                inlet_velocity,
                temperature,
                diameter,
                valve_opening,
                pinn_epochs=fallback_pinn_epochs,
                auto_fluent=auto_fluent,
            )
            scalar = result["scalar"]
            prediction = scalar["predicted_pressure"]
            st.metric("Predicted Max Pressure", f"{prediction:,.1f} Pa")
            st.metric("Status", scalar["status"])
            st.metric("Uncertainty", f"{scalar['uncertainty']:.2%}")

            detail_cols = st.columns(3)
            detail_cols[0].metric("Raw ML Pressure", f"{scalar['predicted_pressure_raw']:,.1f} Pa")
            detail_cols[1].metric("Feedback Adjustment", f"{scalar['feedback_adjustment']:,.1f} Pa")
            detail_cols[2].metric("Feedback Samples", int(scalar["feedback_samples_used"]))

            if result["fluent_triggered"]:
                st.warning("High uncertainty detected. Fluent fallback ran automatically and models were retrained.")
                st.subheader("Post-Fallback Retraining Metrics")
                st.json(result["retrain_metrics"])
                st.subheader("Initial Prediction Before Feedback Loop")
                st.json(result["scalar_initial"])
            elif scalar["trigger_fluent"]:
                st.warning("High uncertainty detected. Automatic Fluent fallback is currently disabled.")
            else:
                st.info("Prediction stayed inside the calibrated confidence band.")

            fig, ax = plt.subplots(figsize=(8, 2))
            low = system.ml_predictor.threshold_low
            high = system.ml_predictor.threshold_high
            ax.barh([0], [prediction], color="steelblue")
            ax.axvline(low, color="green", linestyle="--", label=f"LOW {low:.0f}")
            ax.axvline(high, color="red", linestyle="--", label=f"HIGH {high:.0f}")
            ax.set_xlim(0, max(prediction * 1.2, high * 1.2))
            ax.set_yticks([])
            ax.set_xlabel("Pressure (Pa)")
            ax.legend()
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

elif page == "Field Prediction":
    st.header("4. Field Prediction (Conditioned PINN)")
    condition_col1, condition_col2 = st.columns(2)
    with condition_col1:
        inlet_velocity = st.slider("Conditioned Inlet Velocity (m/s)", 0.1, 10.0, 2.0, 0.1, key="field_vel")
        temperature = st.slider("Conditioned Temperature (C)", 10.0, 100.0, 25.0, 1.0, key="field_temp")
    with condition_col2:
        diameter = st.slider("Conditioned Diameter (m)", 0.01, 0.5, 0.1, 0.01, key="field_diam")
        valve_opening = st.slider("Conditioned Valve Opening", 0.0, 1.0, 0.5, 0.05, key="field_valve")

    col1, col2, col3 = st.columns(3)
    with col1:
        x_min = st.number_input("X min", 0.0, 10.0, 0.0)
        x_max = st.number_input("X max", 0.0, 10.0, 1.0)
    with col2:
        y_min = st.number_input("Y min", 0.0, 10.0, 0.0)
        y_max = st.number_input("Y max", 0.0, 10.0, 0.5)
    with col3:
        z_min = st.number_input("Z min", 0.0, 10.0, 0.0)
        z_max = st.number_input("Z max", 0.0, 10.0, 0.1)

    resolution = st.slider("Grid Resolution", 8, 64, 16, 8)
    auto_refresh = st.checkbox("Auto-refresh models with Fluent if scalar uncertainty is high", value=True)
    refresh_pinn_epochs = st.slider("PINN epochs for field-side fallback retraining", 10, 2000, 100, 10)

    if st.button("Predict Field"):
        try:
            scalar_gate = system.predict_scalar(inlet_velocity, temperature, diameter, valve_opening)
            if scalar_gate["trigger_fluent"] and auto_refresh:
                with st.spinner("Refreshing models through Fluent fallback before field prediction..."):
                    system.full_predict(
                        inlet_velocity,
                        temperature,
                        diameter,
                        valve_opening,
                        pinn_epochs=refresh_pinn_epochs,
                        auto_fluent=True,
                    )

            with st.spinner("Running conditioned PINN on a grid..."):
                grid = system.predict_field_grid(
                    (x_min, x_max),
                    (y_min, y_max),
                    (z_min, z_max),
                    inlet_velocity,
                    temperature,
                    diameter,
                    valve_opening,
                    resolution=resolution,
                )

            st.success("Field prediction complete.")
            mid_z = resolution // 2
            pressure_slice = grid["p"][:, :, mid_z]
            velocity_slice = grid["u"][:, :, mid_z]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.heatmap(pressure_slice, ax=axes[0], cmap="viridis", cbar_kws={"label": "Pressure (Pa)"})
            axes[0].set_title("Pressure (mid Z)")
            axes[0].set_xlabel("Y index")
            axes[0].set_ylabel("X index")

            sns.heatmap(velocity_slice, ax=axes[1], cmap="coolwarm", cbar_kws={"label": "U velocity (m/s)"})
            axes[1].set_title("U Velocity (mid Z)")
            axes[1].set_xlabel("Y index")
            axes[1].set_ylabel("X index")
            st.pyplot(fig)

            st.caption(
                f"Scalar gate uncertainty for this operating condition: {scalar_gate['uncertainty']:.2%}"
            )

            output_path = os.path.join(tempfile.gettempdir(), "field_prediction.npz")
            np.savez(output_path, **grid)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Field (.npz)",
                    data=file,
                    file_name="field_prediction.npz",
                    mime="application/octet-stream",
                )
        except Exception as exc:
            st.error(f"Field prediction error: {exc}")

elif page == "Fluent Fallback":
    st.header("5. Fluent Fallback")
    col1, col2 = st.columns(2)
    with col1:
        f_vel = st.slider("Inlet Velocity", 0.1, 10.0, 2.5, 0.1)
        f_temp = st.slider("Temperature", 10.0, 100.0, 30.0, 1.0)
    with col2:
        f_diam = st.slider("Diameter", 0.01, 0.5, 0.1, 0.01)
        f_valve = st.slider("Valve Opening", 0.0, 1.0, 0.5, 0.05)
    auto_retrain_after_fluent = st.checkbox("Retrain immediately after Fluent run", value=True)
    fluent_pinn_epochs = st.slider("PINN epochs for post-Fluent retraining", 10, 2000, 100, 10)

    if st.button("Run Fluent Simulation"):
        try:
            prediction_context = None
            if system.state.get("ml_trained"):
                prediction_context = system.predict_scalar(f_vel, f_temp, f_diam, f_valve)

            with st.spinner("Running Fluent (or mock fallback)..."):
                df = system.run_fluent_fallback(
                    f_vel,
                    f_temp,
                    f_diam,
                    f_valve,
                    prediction_context=prediction_context,
                    feedback_source="manual_dashboard_fluent",
                )
            st.success("Fluent fallback complete.")
            st.dataframe(df.head(10), use_container_width=True)

            if auto_retrain_after_fluent:
                with st.spinner("Retraining ML and PINN..."):
                    metrics = system.retrain_all(pinn_epochs=fluent_pinn_epochs)
                st.subheader("Retraining Metrics")
                st.json(metrics)
        except Exception as exc:
            st.error(f"Fluent fallback error: {exc}")

elif page == "System Status":
    st.header("6. System Status")
    st.subheader("Data Summary")
    st.json(get_summary())
    st.subheader("Pipeline State")
    st.json(system.state)
    st.subheader("Feedback Loop")
    st.json(system.get_feedback_summary())
    st.subheader("Automation")
    st.json(get_automation_summary())

    if system.state.get("ml_trained"):
        st.success("ML model: trained")
    else:
        st.warning("ML model: not trained")

    if system.state.get("pinn_trained"):
        st.success("PINN model: trained")
    else:
        st.warning("PINN model: not trained")
