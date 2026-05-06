# Closed-Loop Hybrid AI-CFD Workflow

A hybrid aerodynamic prediction framework combining:

- Fast machine learning surrogate inference
- Physics-Informed Neural Networks (PINNs)
- Uncertainty-aware decision routing
- High-fidelity CFD validation using Fluent
- Continuous feedback retraining

---

## Overview

Traditional CFD simulations are computationally expensive and difficult to scale for rapid inference workflows.

This project explores a closed-loop AI-CFD architecture where:

- low-latency ML models provide fast predictions,
- uncertainty estimation determines prediction confidence,
- high-uncertainty cases are validated using Fluent CFD simulations,
- validated results are logged and continuously used for adaptive retraining.

The system combines fast inference with high-fidelity validation to create a continuously improving aerodynamic prediction pipeline.

---

## System Architecture

![Workflow](workflow.png)

---

## Core Components

### Fast ML Predictor
Performs low-latency aerodynamic scalar prediction using trained surrogate models.

### Uncertainty Estimator
Evaluates model confidence and determines whether predictions require CFD validation.

### Fluent CFD Validation
Runs high-fidelity CFD simulations for uncertain operating conditions using PyFluent integration.

### PINN Field Prediction
Generates velocity and pressure field estimations using Physics-Informed Neural Networks conditioned on spatial coordinates and operating parameters.

### Feedback Logging
Stores validated CFD outcomes and prediction metadata for continuous learning.

### Model Retraining
Automatically updates prediction and uncertainty models using accumulated feedback data.

---

## Features

- Closed-loop AI-CFD workflow
- Physics-informed neural networks (PINNs)
- Uncertainty-aware inference pipeline
- Automated CFD fallback
- Continuous retraining system
- Streamlit dashboard support
- PyFluent integration hooks
- Automated workflow scheduling

---

## Tech Stack

- Python
- PyTorch
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- PyFluent
- Matplotlib

---

## Running the System

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run dashboard.py
```

### Run Automation Scheduler

```bash
python automation.py
```

---

## Project Structure

```text
├── automation.py          # Background scheduling pipeline
├── dashboard.py           # Streamlit dashboard
├── data_handler.py        # Data validation and ingestion
├── feedback_loop.py       # Feedback logging and calibration
├── fluent_wrapper.py      # PyFluent integration
├── ml_module.py           # Surrogate ML models
├── pinn_module.py         # PINN implementation
├── pipeline.py            # Unified workflow pipeline
├── workflow.png           # System architecture diagram
└── requirements.txt
```

---

## Future Work

- Multi-geometry support
- GPU-accelerated PINN training
- OpenFOAM integration
- Active learning for CFD sampling
- Advanced uncertainty calibration
- Distributed simulation orchestration

---

## Status

Research engineering prototype for hybrid AI-assisted CFD workflows.

Real Fluent execution requires:
- Ansys Fluent installation
- Valid Fluent license
- Configured boundary-condition mappings
- Proper case/data setup

---
