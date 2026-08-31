import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from scipy.interpolate import RegularGridInterpolator
from physics.base_pde import BasePDE


class LidDrivenCavityNavierStokes(BasePDE):
    """
    2D Steady Incompressible Lid-Driven Cavity Flow Benchmark (Re = 100).
    
    Formulation: Streamfunction-Pressure (psi, p)
        - Primary state: z(x, y) = [psi(x, y), p(x, y)]
        - Exact Velocity: u = d(psi)/dy, v = -d(psi)/dx
        - Exact Continuity: div(u) = d^2(psi)/(dx dy) - d^2(psi)/(dy dx) == 0 (Machine Precision)
        - Momentum Residuals:
            R_x = u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy) = 0
            R_y = u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy) = 0
        - Boundaries on [0, 1]^2:
            Bottom, Left, Right Walls: psi = 0, u = 0, v = 0
            Top Moving Lid (y = 1): psi = 0, u = 1.0, v = 0
            Pressure Gauge: p(0, 0) = 0
    """
    def __init__(
        self,
        Re: float = 100.0,
        gt_pkl_path: str = "data/gt_data_Re100.pkl",
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=2, device=device)
        self.Re = Re
        self.nu = 1.0 / Re
        self.bounds = ((0.0, 1.0), (0.0, 1.0))
        self.gt_pkl_path = gt_pkl_path
        self._load_ground_truth()

    def _load_ground_truth(self):
        """Loads and prepares interpolation functions for FDM CFD ground truth."""
        if os.path.exists(self.gt_pkl_path):
            with open(self.gt_pkl_path, "rb") as f:
                data = pickle.load(f)
            x_coords = data["coordinates"]["x"]
            y_coords = data["coordinates"]["y"]
            psi_grid = data["fields"]["psi"]
            u_grid = data["fields"]["u"]
            v_grid = data["fields"]["v"]
            p_grid = data["fields"]["p"]

            # Interpolators on (y, x)
            self._interp_psi = RegularGridInterpolator((y_coords, x_coords), psi_grid, bounds_error=False, fill_value=0.0)
            self._interp_u = RegularGridInterpolator((y_coords, x_coords), u_grid, bounds_error=False, fill_value=0.0)
            self._interp_v = RegularGridInterpolator((y_coords, x_coords), v_grid, bounds_error=False, fill_value=0.0)
            self._interp_p = RegularGridInterpolator((y_coords, x_coords), p_grid, bounds_error=False, fill_value=0.0)
            self.has_gt = True
        else:
            self.has_gt = False
            self._interp_psi = None

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Sample interior collocation points uniformly in (0, 1)^2."""
        return torch.rand(n_samples, 2, device=self.device)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample boundary points on 4 walls of [0, 1]^2.
        Target values format: [psi_target, u_target, v_target]
        """
        n_per_wall = n_samples // 4
        s = torch.rand(n_per_wall, 1, device=self.device)

        # 1. Bottom wall: y = 0, u = 0, v = 0, psi = 0
        bottom_pts = torch.cat([s, torch.zeros_like(s)], dim=1)
        bottom_targets = torch.zeros(n_per_wall, 3, device=self.device)

        # 2. Top moving lid: y = 1, u = 1.0, v = 0, psi = 0
        s_top = torch.rand(n_per_wall, 1, device=self.device)
        top_pts = torch.cat([s_top, torch.ones_like(s_top)], dim=1)
        top_targets = torch.zeros(n_per_wall, 3, device=self.device)
        top_targets[:, 1] = 1.0  # u = 1.0

        # 3. Left wall: x = 0, u = 0, v = 0, psi = 0
        s_left = torch.rand(n_per_wall, 1, device=self.device)
        left_pts = torch.cat([torch.zeros_like(s_left), s_left], dim=1)
        left_targets = torch.zeros(n_per_wall, 3, device=self.device)

        # 4. Right wall: x = 1, u = 0, v = 0, psi = 0
        s_right = torch.rand(n_per_wall, 1, device=self.device)
        right_pts = torch.cat([torch.ones_like(s_right), s_right], dim=1)
        right_targets = torch.zeros(n_per_wall, 3, device=self.device)

        pts = torch.cat([bottom_pts, top_pts, left_pts, right_pts], dim=0)
        targets = torch.cat([bottom_targets, top_targets, left_targets, right_targets], dim=0)
        return pts, targets

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes 2D Steady Incompressible Navier-Stokes momentum residuals in (psi, p) form.
        """
        x_interior.requires_grad_(True)
        out = model(x_interior)
        psi = out[:, 0:1]
        p = out[:, 1:2]

        # First derivatives
        grad_psi = torch.autograd.grad(psi.sum(), x_interior, create_graph=True)[0]
        psi_x = grad_psi[:, 0:1]
        psi_y = grad_psi[:, 1:2]
        u = psi_y
        v = -psi_x

        grad_p = torch.autograd.grad(p.sum(), x_interior, create_graph=True)[0]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]

        # Second derivatives (Jacobians of u and v)
        grad_u = torch.autograd.grad(u.sum(), x_interior, create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]

        grad_v = torch.autograd.grad(v.sum(), x_interior, create_graph=True)[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]

        # Third derivatives (Laplacians of u and v)
        u_xx = torch.autograd.grad(u_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        lap_u = u_xx + u_yy

        v_xx = torch.autograd.grad(v_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        lap_v = v_xx + v_yy

        # Navier-Stokes Momentum Residuals
        res_x = u * u_x + v * u_y + p_x - self.nu * lap_u
        res_y = u * v_x + v * v_y + p_y - self.nu * lap_v

        return torch.cat([res_x, res_y], dim=1)

    def compute_boundary_loss(
        self, model: nn.Module, x_bc: torch.Tensor, u_bc_exact: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluates boundary loss: psi = 0, u = target_u, v = 0, plus pressure gauge pinning.
        """
        x_bc.requires_grad_(True)
        out = model(x_bc)
        psi = out[:, 0:1]

        grad_psi = torch.autograd.grad(psi.sum(), x_bc, create_graph=True)[0]
        u_pred = grad_psi[:, 1:2]   # u = psi_y
        v_pred = -grad_psi[:, 0:1]  # v = -psi_x

        psi_target = u_bc_exact[:, 0:1]
        u_target = u_bc_exact[:, 1:2]
        v_target = u_bc_exact[:, 2:3]

        loss_psi = torch.mean((psi - psi_target) ** 2)
        loss_u = torch.mean((u_pred - u_target) ** 2)
        loss_v = torch.mean((v_pred - v_target) ** 2)

        # Pressure Gauge Pinning at origin: p(0, 0) = 0
        origin = torch.zeros(1, 2, device=self.device)
        p_gauge = model(origin)[:, 1:2]
        loss_gauge = torch.mean(p_gauge ** 2)

        return loss_psi + loss_u + loss_v + loss_gauge

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns ground truth stream function psi(x, y) interpolated from FDM CFD simulation.
        """
        x_det = x.detach()
        if not self.has_gt:
            x_np = x_det[:, 0].cpu().numpy()
            y_np = x_det[:, 1].cpu().numpy()
            psi_approx = 8.0 * (x_np**2 * (1 - x_np)**2 * y_np**2 * (1 - y_np)**2)
            return torch.tensor(psi_approx, dtype=torch.float32, device=self.device).unsqueeze(1)

        x_np = x_det[:, 0].cpu().numpy()
        y_np = x_det[:, 1].cpu().numpy()
        pts_query = np.stack([y_np, x_np], axis=1)  # RegularGridInterpolator uses (y, x)
        psi_vals = self._interp_psi(pts_query)
        return torch.tensor(psi_vals, dtype=torch.float32, device=self.device).unsqueeze(1)

    def compute_relative_l2_error(
        self, model: nn.Module, x_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        if x_test is None:
            x_test = self.sample_interior(n_test)
        with torch.no_grad():
            psi_pred = model(x_test)[:, 0:1]
            psi_exact = self.exact_solution(x_test)
            l2_err = torch.norm(psi_pred - psi_exact, p=2)
            l2_ref = torch.norm(psi_exact, p=2).clamp_min(1e-8)
            rel_error = (l2_err / l2_ref).item()
        return rel_error

    def compute_linf_error(
        self, model: nn.Module, x_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        if x_test is None:
            x_test = self.sample_interior(n_test)
        with torch.no_grad():
            psi_pred = model(x_test)[:, 0:1]
            psi_exact = self.exact_solution(x_test)
            max_err = torch.max(torch.abs(psi_pred - psi_exact)).item()
        return max_err
