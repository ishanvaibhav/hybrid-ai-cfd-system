"""
Physics-informed neural network for CFD field prediction.

Input:
    spatial coordinates plus operating conditions
Output:
    velocity components (u, v, w) and pressure (p)
Loss:
    data fidelity + incompressible Navier-Stokes residuals
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PINN_MODEL_PATH = os.path.join(MODEL_DIR, "pinn_model.pth")
PINN_META_PATH = os.path.join(MODEL_DIR, "pinn_meta.json")

DEFAULT_RHO = 1000.0
DEFAULT_MU = 1.0e-3

COORD_COLS = ["x", "y", "z"]
CONDITION_COLS = ["inlet_velocity", "temperature", "diameter", "valve_opening"]
PINN_INPUT_COLS = COORD_COLS + CONDITION_COLS
FIELD_COLS = ["u", "v", "w", "p"]


def _ensure_dirs() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


class NavierStokesPINN(nn.Module):
    def __init__(self, layers: Optional[List[int]] = None, activation: str = "tanh") -> None:
        super().__init__()
        if layers is None:
            layers = [len(PINN_INPUT_COLS), 96, 96, 96, 96, len(FIELD_COLS)]

        self.layers = layers
        self.activation_name = activation

        modules: List[nn.Module] = []
        for index in range(len(layers) - 1):
            modules.append(nn.Linear(layers[index], layers[index + 1]))
            if index < len(layers) - 2:
                modules.append(nn.Tanh() if activation == "tanh" else nn.ReLU())

        self.net = nn.Sequential(*modules)
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def compute_derivatives(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    gradients = []
    for index in range(y.shape[1]):
        derivative = grad(
            y[:, index],
            x,
            grad_outputs=torch.ones_like(y[:, index]),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients.append(derivative)
    return torch.stack(gradients, dim=1)


def navier_stokes_residual(
    spatial_coords: torch.Tensor,
    prediction: torch.Tensor,
    rho: float = DEFAULT_RHO,
    mu: float = DEFAULT_MU,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u = prediction[:, 0:1]
    v = prediction[:, 1:2]
    w = prediction[:, 2:3]
    p = prediction[:, 3:4]

    first_derivatives = compute_derivatives(torch.cat([u, v, w, p], dim=1), spatial_coords)

    u_x, u_y, u_z = first_derivatives[:, 0, 0], first_derivatives[:, 0, 1], first_derivatives[:, 0, 2]
    v_x, v_y, v_z = first_derivatives[:, 1, 0], first_derivatives[:, 1, 1], first_derivatives[:, 1, 2]
    w_x, w_y, w_z = first_derivatives[:, 2, 0], first_derivatives[:, 2, 1], first_derivatives[:, 2, 2]
    p_x, p_y, p_z = first_derivatives[:, 3, 0], first_derivatives[:, 3, 1], first_derivatives[:, 3, 2]

    continuity = u_x + v_y + w_z

    u_xx = grad(u_x, spatial_coords, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = grad(u_y, spatial_coords, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    u_zz = grad(u_z, spatial_coords, grad_outputs=torch.ones_like(u_z), create_graph=True, retain_graph=True)[0][:, 2:3]

    v_xx = grad(v_x, spatial_coords, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    v_yy = grad(v_y, spatial_coords, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    v_zz = grad(v_z, spatial_coords, grad_outputs=torch.ones_like(v_z), create_graph=True, retain_graph=True)[0][:, 2:3]

    w_xx = grad(w_x, spatial_coords, grad_outputs=torch.ones_like(w_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    w_yy = grad(w_y, spatial_coords, grad_outputs=torch.ones_like(w_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    w_zz = grad(w_z, spatial_coords, grad_outputs=torch.ones_like(w_z), create_graph=True, retain_graph=True)[0][:, 2:3]

    lap_u = u_xx + u_yy + u_zz
    lap_v = v_xx + v_yy + v_zz
    lap_w = w_xx + w_yy + w_zz

    conv_u = u * u_x.unsqueeze(1) + v * u_y.unsqueeze(1) + w * u_z.unsqueeze(1)
    conv_v = u * v_x.unsqueeze(1) + v * v_y.unsqueeze(1) + w * v_z.unsqueeze(1)
    conv_w = u * w_x.unsqueeze(1) + v * w_y.unsqueeze(1) + w * w_z.unsqueeze(1)

    momentum_x = rho * conv_u + p_x.unsqueeze(1) - mu * lap_u
    momentum_y = rho * conv_v + p_y.unsqueeze(1) - mu * lap_v
    momentum_z = rho * conv_w + p_z.unsqueeze(1) - mu * lap_w

    loss_continuity = torch.mean(continuity ** 2)
    loss_mx = torch.mean(momentum_x ** 2)
    loss_my = torch.mean(momentum_y ** 2)
    loss_mz = torch.mean(momentum_z ** 2)
    return loss_continuity, loss_mx, loss_my, loss_mz


class PINNTrainer:
    def __init__(
        self,
        rho: float = DEFAULT_RHO,
        mu: float = DEFAULT_MU,
        layers: Optional[List[int]] = None,
        lr: float = 1.0e-3,
        device: Optional[str] = None,
        activation: str = "tanh",
    ) -> None:
        self.rho = rho
        self.mu = mu
        self.layers = layers or [len(PINN_INPUT_COLS), 96, 96, 96, 96, len(FIELD_COLS)]
        self.lr = lr
        self.activation = activation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.5)
        self.loss_history: List[Dict[str, float]] = []
        self.is_trained = False

    def _build_model(self) -> NavierStokesPINN:
        return NavierStokesPINN(layers=self.layers, activation=self.activation).to(self.device)

    def _rebuild_training_objects(self) -> None:
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.5)

    def required_training_columns(self) -> List[str]:
        return PINN_INPUT_COLS

    def _compose_inputs(
        self,
        coordinates: np.ndarray,
        operating_conditions: Dict[str, float],
    ) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(COORD_COLS):
            raise ValueError("Coordinates must be a 2D array with shape (n_points, 3).")

        condition_vector = np.array(
            [float(operating_conditions[column]) for column in CONDITION_COLS],
            dtype=np.float32,
        )
        tiled_conditions = np.repeat(condition_vector.reshape(1, -1), len(coordinates), axis=0)
        return np.concatenate([coordinates, tiled_conditions], axis=1)

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 5000,
        lambda_data: float = 1.0,
        lambda_phys: float = 1.0,
        verbose_interval: int = 500,
    ) -> Dict[str, Any]:
        missing_inputs = [column for column in PINN_INPUT_COLS if column not in df.columns]
        if missing_inputs:
            raise ValueError(f"PINN training requires conditioned columns: {missing_inputs}")

        targets: Dict[str, Optional[torch.Tensor]] = {name: None for name in FIELD_COLS}
        has_supervision = False
        for key in FIELD_COLS:
            if key in df.columns and df[key].notna().any():
                targets[key] = torch.tensor(df[key].values.astype(np.float32), dtype=torch.float32, device=self.device)
                has_supervision = True

        if not has_supervision:
            raise ValueError("PINN training requires at least one supervised field column: u, v, w, or p.")

        self.model.train()

        input_np = df[PINN_INPUT_COLS].values.astype(np.float32)
        model_inputs = torch.tensor(input_np, dtype=torch.float32, device=self.device)
        spatial_coords = model_inputs[:, : len(COORD_COLS)].clone().detach().requires_grad_(True)
        condition_features = model_inputs[:, len(COORD_COLS):].clone().detach()

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            conditioned_inputs = torch.cat([spatial_coords, condition_features], dim=1)
            prediction = self.model(conditioned_inputs)

            loss_data = torch.zeros((), dtype=torch.float32, device=self.device)
            if targets["u"] is not None:
                loss_data = loss_data + torch.mean((prediction[:, 0] - targets["u"]) ** 2)
            if targets["v"] is not None:
                loss_data = loss_data + torch.mean((prediction[:, 1] - targets["v"]) ** 2)
            if targets["w"] is not None:
                loss_data = loss_data + torch.mean((prediction[:, 2] - targets["w"]) ** 2)
            if targets["p"] is not None:
                loss_data = loss_data + torch.mean((prediction[:, 3] - targets["p"]) ** 2)

            loss_c, loss_mx, loss_my, loss_mz = navier_stokes_residual(spatial_coords, prediction, self.rho, self.mu)
            loss_physics = loss_c + loss_mx + loss_my + loss_mz
            loss = lambda_data * loss_data + lambda_phys * loss_physics

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.loss_history.append(
                {
                    "epoch": float(epoch),
                    "total": float(loss.item()),
                    "data": float(loss_data.item()),
                    "physics": float(loss_physics.item()),
                    "continuity": float(loss_c.item()),
                }
            )

            if verbose_interval and epoch % verbose_interval == 0:
                print(
                    f"Epoch {epoch:5d} | Total: {loss.item():.3e} | "
                    f"Data: {loss_data.item():.3e} | Phys: {loss_physics.item():.3e} | "
                    f"Cont: {loss_c.item():.3e}"
                )

        self.is_trained = True
        self.save()
        return {
            "final_loss": float(loss.item()),
            "final_data_loss": float(loss_data.item()),
            "final_physics_loss": float(loss_physics.item()),
            "epochs": int(epochs),
            "device": self.device,
            "input_columns": PINN_INPUT_COLS,
        }

    def predict_field(
        self,
        coordinates: np.ndarray,
        operating_conditions: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("PINN is not trained yet.")

        inputs = self._compose_inputs(coordinates, operating_conditions)
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            prediction = self.model(input_tensor).cpu().numpy()

        return {
            "u": prediction[:, 0],
            "v": prediction[:, 1],
            "w": prediction[:, 2],
            "p": prediction[:, 3],
        }

    def predict_grid(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        operating_conditions: Dict[str, float],
        resolution: int = 32,
    ) -> Dict[str, np.ndarray]:
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        coordinates = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        field = self.predict_field(coordinates, operating_conditions)
        shape = (resolution, resolution, resolution)
        return {
            "x": xx,
            "y": yy,
            "z": zz,
            "u": field["u"].reshape(shape),
            "v": field["v"].reshape(shape),
            "w": field["w"].reshape(shape),
            "p": field["p"].reshape(shape),
        }

    def save(self) -> None:
        _ensure_dirs()
        torch.save(self.model.state_dict(), PINN_MODEL_PATH)
        meta = {
            "rho": self.rho,
            "mu": self.mu,
            "layers": self.layers,
            "lr": self.lr,
            "device": self.device,
            "activation": self.activation,
            "input_columns": PINN_INPUT_COLS,
            "loss_history_last": self.loss_history[-1] if self.loss_history else None,
        }
        with open(PINN_META_PATH, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

    def load(self) -> None:
        if not os.path.exists(PINN_MODEL_PATH):
            raise FileNotFoundError("No saved PINN model found. Train first.")

        if os.path.exists(PINN_META_PATH):
            with open(PINN_META_PATH, "r", encoding="utf-8") as file:
                meta = json.load(file)
            self.rho = meta.get("rho", DEFAULT_RHO)
            self.mu = meta.get("mu", DEFAULT_MU)
            self.layers = meta.get("layers", self.layers)
            self.lr = meta.get("lr", self.lr)
            self.activation = meta.get("activation", self.activation)
            self._rebuild_training_objects()

        self.model.load_state_dict(torch.load(PINN_MODEL_PATH, map_location=self.device))
        self.is_trained = True


def get_trained_pinn() -> PINNTrainer:
    trainer = PINNTrainer()
    try:
        trainer.load()
    except FileNotFoundError:
        pass
    return trainer
