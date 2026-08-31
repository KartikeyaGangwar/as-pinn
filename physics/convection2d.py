"""
2D High-Péclet Convection-Dominated Transport Benchmark (Krishnapriyan Failure Mode)
Exact Analytical Transport of 2D Waves with High Advection Velocity.
Governing Equation:
    u_t + beta_x * u_x + beta_y * u_y - nu * (u_xx + u_yy) = 0
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from physics.base_pde import BasePDE


class HighPecletConvection2D(BasePDE):
    def __init__(
        self,
        beta_x: float = 10.0,
        beta_y: float = 10.0,
        nu: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=3, out_dim=1, device=device) # (x, y, t) -> u
        self.beta_x = float(beta_x)
        self.beta_y = float(beta_y)
        self.nu = float(nu)
        
        # Spatial-temporal domain: [0, 2pi]^2 x [0, 1.0]
        self.bounds = ((0.0, float(2.0 * np.pi)), (0.0, float(2.0 * np.pi)), (0.0, 1.0))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact Analytical Solution for 2D Convection-Diffusion:
        u(x, y, t) = sin(x - beta_x * t) * sin(y - beta_y * t) * exp(-2 * nu * t)
        """
        x_pos = x[:, 0:1]
        y_pos = x[:, 1:2]
        t_pos = x[:, 2:3]
        
        phase_x = x_pos - self.beta_x * t_pos
        phase_y = y_pos - self.beta_y * t_pos
        decay = torch.exp(-2.0 * self.nu * t_pos)
        
        u = torch.sin(phase_x) * torch.sin(phase_y) * decay
        return u

    def sample_interior(self, n_points: int) -> torch.Tensor:
        two_pi = float(2.0 * np.pi)
        x = torch.rand(n_points, 1, device=self.device) * two_pi
        y = torch.rand(n_points, 1, device=self.device) * two_pi
        t = torch.rand(n_points, 1, device=self.device) * 1.0
        return torch.cat([x, y, t], dim=1)

    def sample_boundary(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_side = n_points // 4
        two_pi = float(2.0 * np.pi)
        
        # Side 1: x = 0, y in [0, 2pi], t in [0, 1]
        x1 = torch.zeros((n_side, 1), device=self.device)
        y1 = torch.rand(n_side, 1, device=self.device) * two_pi
        t1 = torch.rand(n_side, 1, device=self.device) * 1.0
        b1 = torch.cat([x1, y1, t1], dim=1)
        
        # Side 2: x = 2pi, y in [0, 2pi], t in [0, 1]
        x2 = torch.full((n_side, 1), two_pi, device=self.device)
        y2 = torch.rand(n_side, 1, device=self.device) * two_pi
        t2 = torch.rand(n_side, 1, device=self.device) * 1.0
        b2 = torch.cat([x2, y2, t2], dim=1)
        
        # Side 3: y = 0, x in [0, 2pi], t in [0, 1]
        x3 = torch.rand(n_side, 1, device=self.device) * two_pi
        y3 = torch.zeros((n_side, 1), device=self.device)
        t3 = torch.rand(n_side, 1, device=self.device) * 1.0
        b3 = torch.cat([x3, y3, t3], dim=1)
        
        # Side 4: y = 2pi, x in [0, 2pi], t in [0, 1]
        x4 = torch.rand(n_side, 1, device=self.device) * two_pi
        y4 = torch.full((n_side, 1), two_pi, device=self.device)
        t4 = torch.rand(n_side, 1, device=self.device) * 1.0
        b4 = torch.cat([x4, y4, t4], dim=1)
        
        pts = torch.cat([b1, b2, b3, b4], dim=0)
        target = self.exact_solution(pts)
        return pts, target

    def sample_initial(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        two_pi = float(2.0 * np.pi)
        x = torch.rand(n_points, 1, device=self.device) * two_pi
        y = torch.rand(n_points, 1, device=self.device) * two_pi
        t = torch.zeros((n_points, 1), device=self.device)
        
        pts = torch.cat([x, y, t], dim=1)
        target = self.exact_solution(pts)
        return pts, target

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        x = x_interior.clone().detach().requires_grad_(True)
        u = model(x)
        if u.shape[1] > 1:
            u = u[:, 0:1]

        grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]

        # Convection-Diffusion Residual: u_t + beta_x * u_x + beta_y * u_y - nu * (u_xx + u_yy) = 0
        res = u_t + self.beta_x * u_x + self.beta_y * u_y - self.nu * (u_xx + u_yy)
        return res

    def pde_residual(self, model, x: torch.Tensor) -> torch.Tensor:
        return self.compute_residuals(model, x)
