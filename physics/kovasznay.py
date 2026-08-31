"""
2D Steady Incompressible Kovasznay Flow Benchmark (Navier-Stokes Re=40)
Exact Analytical Laminar Wake behind a periodic cylinder grid.
Governing Equations:
    u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy) = 0
    u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy) = 0
    u_x + v_y = 0
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from physics.base_pde import BasePDE


class KovasznayFlow2D(BasePDE):
    def __init__(self, Re: float = 40.0, device: Optional[torch.device] = None):
        super().__init__(in_dim=2, out_dim=3, device=device)
        self.Re = float(Re)
        self.nu = 1.0 / self.Re
        
        # Spatial domain: [-0.5, 1.0] x [-0.5, 1.5]
        self.bounds = ((-0.5, 1.0), (-0.5, 1.5))
        
        # Kovasznay parameter lambda
        # lambda = Re/2 - sqrt(Re^2/4 + 4*pi^2)
        self.lam = float(self.Re / 2.0 - np.sqrt(self.Re ** 2 / 4.0 + 4.0 * (np.pi ** 2)))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact Analytical Solution for Kovasznay Flow:
        u(x, y) = 1 - exp(lam * x) * cos(2*pi*y)
        v(x, y) = (lam / (2*pi)) * exp(lam * x) * sin(2*pi*y)
        p(x, y) = -0.5 * exp(2 * lam * x)
        """
        x_pos = x[:, 0:1]
        y_pos = x[:, 1:2]
        
        lam = self.lam
        two_pi = 2.0 * np.pi
        
        exp_lx = torch.exp(lam * x_pos)
        
        u = 1.0 - exp_lx * torch.cos(two_pi * y_pos)
        v = (lam / two_pi) * exp_lx * torch.sin(two_pi * y_pos)
        p = -0.5 * torch.exp(2.0 * lam * x_pos)
        
        return torch.cat([u, v, p], dim=1)

    def sample_interior(self, n_points: int) -> torch.Tensor:
        x1 = torch.rand(n_points, 1, device=self.device) * 1.5 - 0.5   # [-0.5, 1.0]
        x2 = torch.rand(n_points, 1, device=self.device) * 2.0 - 0.5   # [-0.5, 1.5]
        return torch.cat([x1, x2], dim=1)

    def sample_boundary(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_side = n_points // 4
        # Side 1: x = -0.5, y in [-0.5, 1.5]
        y1 = torch.rand(n_side, 1, device=self.device) * 2.0 - 0.5
        x1 = torch.full((n_side, 1), -0.5, device=self.device)
        b1 = torch.cat([x1, y1], dim=1)
        
        # Side 2: x = 1.0, y in [-0.5, 1.5]
        y2 = torch.rand(n_side, 1, device=self.device) * 2.0 - 0.5
        x2 = torch.full((n_side, 1), 1.0, device=self.device)
        b2 = torch.cat([x2, y2], dim=1)
        
        # Side 3: y = -0.5, x in [-0.5, 1.0]
        x3 = torch.rand(n_side, 1, device=self.device) * 1.5 - 0.5
        y3 = torch.full((n_side, 1), -0.5, device=self.device)
        b3 = torch.cat([x3, y3], dim=1)
        
        # Side 4: y = 1.5, x in [-0.5, 1.0]
        x4 = torch.rand(n_side, 1, device=self.device) * 1.5 - 0.5
        y4 = torch.full((n_side, 1), 1.5, device=self.device)
        b4 = torch.cat([x4, y4], dim=1)
        
        pts = torch.cat([b1, b2, b3, b4], dim=0)
        target = self.exact_solution(pts)
        return pts, target

    def sample_initial(self, n_points: int):
        return None

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        x = x_interior.clone().detach().requires_grad_(True)
        out = model(x)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]

        grad_v = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]

        grad_p = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]

        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]

        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:, 1:2]

        # Incompressible Navier-Stokes residuals
        res_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        res_c = u_x + v_y  # Continuity / Divergence free

        return torch.cat([res_u, res_v, res_c], dim=1)

    def pde_residual(self, model, x: torch.Tensor) -> torch.Tensor:
        return self.compute_residuals(model, x)
