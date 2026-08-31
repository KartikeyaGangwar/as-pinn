import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE


class NonLinearBratu2D(BasePDE):
    """
    2D Non-Linear Bratu / Gelfand Thermal Ignition Equation:
        -Delta u - lambda * exp(u) = f(x, y)
        
    on (x, y) in [-1, 1]^2 with parameter lambda = 1.0.
    
    Models thermonuclear runaway, plasma equilibrium, and solid fuel combustion
    where intense localized exponential nonlinearity creates severe gradient contention.
    
    Exact Analytical Localized Thermal Core:
        u*(x, y) = 2.0 * exp(-3.0 * (x^2 + y^2))
    """
    def __init__(
        self,
        lambda_param: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=1, device=device)
        self.lambda_param = lambda_param
        self.bounds = ((-1.0, 1.0), (-1.0, 1.0))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Returns exact thermal ignition field u*(x, y)."""
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        r_sq = x_c**2 + y_c**2
        return 2.0 * torch.exp(-3.0 * r_sq)

    def _forcing(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates exact RHS forcing function f(x, y) = -Delta u* - lambda * exp(u*)."""
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        r_sq = x_c**2 + y_c**2
        u_star = 2.0 * torch.exp(-3.0 * r_sq)

        # Derivatives of u* = 2 * exp(-3 r^2):
        # u_x = -12 x exp(-3 r^2)
        # u_xx = (-12 + 72 x^2) exp(-3 r^2)
        # u_yy = (-12 + 72 y^2) exp(-3 r^2)
        # Delta u = (-24 + 72 r^2) exp(-3 r^2)
        lap_u = (-24.0 + 72.0 * r_sq) * torch.exp(-3.0 * r_sq)

        f = -lap_u - self.lambda_param * torch.exp(u_star)
        return f

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Sample interior collocation points in (-1, 1)^2."""
        return torch.empty(n_samples, 2, device=self.device).uniform_(-1.0, 1.0)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample Dirichlet boundary points on 4 walls of [-1, 1]^2."""
        n_per_wall = n_samples // 4
        s = torch.empty(n_per_wall, 1, device=self.device).uniform_(-1.0, 1.0)

        b1 = torch.cat([s, torch.full_like(s, -1.0)], dim=1)
        b2 = torch.cat([s, torch.full_like(s, 1.0)], dim=1)
        b3 = torch.cat([torch.full_like(s, -1.0), s], dim=1)
        b4 = torch.cat([torch.full_like(s, 1.0), s], dim=1)

        pts = torch.cat([b1, b2, b3, b4], dim=0)
        u_bc = self.exact_solution(pts)
        return pts, u_bc

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes Bratu residual:
            R = -Delta u - lambda * exp(u) - f(x, y)
        """
        x_interior.requires_grad_(True)
        u = model(x_interior)

        grad_u = torch.autograd.grad(u.sum(), x_interior, create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]

        u_xx = torch.autograd.grad(u_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        lap_u = u_xx + u_yy

        f = self._forcing(x_interior)

        # Clamping u inside exp to prevent numerical overflow in early random initialization
        u_clamped = torch.clamp(u, max=15.0)
        res = -lap_u - self.lambda_param * torch.exp(u_clamped) - f
        return res
