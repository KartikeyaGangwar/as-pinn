import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE


class KleinGordon2D(BasePDE):
    """
    2D Spatio-Temporal Nonlinear Relativistic Klein-Gordon Equation:
        u_tt - c^2 * (u_xx + u_yy) + alpha * u + beta * u^3 = f(x, y, t)
        
    on (x, y, t) in [-1, 1]^2 x [0, 1].
    Parameters: c = 1.0, alpha = 1.0, beta = 1.0.
    
    Analytical Exact Multi-Harmonic Soliton:
        u*(x, y, t) = (x + y) * cos(2*pi*t) + sin(pi*x) * sin(pi*y) * cos(4*pi*t)
    """
    def __init__(
        self,
        c: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=3, out_dim=1, device=device)
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates exact analytical solution u*(x, y, t)."""
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        t_c = x[:, 2:3]
        pi = np.pi

        u1 = (x_c + y_c) * torch.cos(2.0 * pi * t_c)
        u2 = torch.sin(pi * x_c) * torch.sin(pi * y_c) * torch.cos(4.0 * pi * t_c)
        return u1 + u2

    def _forcing(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates exact analytical RHS forcing function f(x, y, t)."""
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        t_c = x[:, 2:3]
        pi = np.pi

        u_star = self.exact_solution(x)
        u_tt = -4.0 * (pi**2) * (x_c + y_c) * torch.cos(2.0 * pi * t_c) - 16.0 * (pi**2) * torch.sin(pi * x_c) * torch.sin(pi * y_c) * torch.cos(4.0 * pi * t_c)
        lap_u = -2.0 * (pi**2) * torch.sin(pi * x_c) * torch.sin(pi * y_c) * torch.cos(4.0 * pi * t_c)

        f = u_tt - (self.c**2) * lap_u + self.alpha * u_star + self.beta * (u_star**3)
        return f

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Sample interior collocation points in (-1, 1)^2 x (0, 1)."""
        xy = torch.empty(n_samples, 2, device=self.device).uniform_(-1.0, 1.0)
        t = torch.empty(n_samples, 1, device=self.device).uniform_(0.0, 1.0)
        return torch.cat([xy, t], dim=1)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample spatial boundary points on dOmega x [0, 1]."""
        n_per_wall = n_samples // 4
        t = torch.empty(n_per_wall, 1, device=self.device).uniform_(0.0, 1.0)
        s = torch.empty(n_per_wall, 1, device=self.device).uniform_(-1.0, 1.0)

        # 4 spatial boundaries:
        b1 = torch.cat([s, torch.full_like(s, -1.0), t], dim=1)
        b2 = torch.cat([s, torch.full_like(s, 1.0), t], dim=1)
        b3 = torch.cat([torch.full_like(s, -1.0), s, t], dim=1)
        b4 = torch.cat([torch.full_like(s, 1.0), s, t], dim=1)

        pts = torch.cat([b1, b2, b3, b4], dim=0)
        u_bc = self.exact_solution(pts)
        return pts, u_bc

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample t = 0 initial condition points."""
        xy = torch.empty(n_samples, 2, device=self.device).uniform_(-1.0, 1.0)
        t0 = torch.zeros(n_samples, 1, device=self.device)
        pts = torch.cat([xy, t0], dim=1)
        u_ic = self.exact_solution(pts)
        return pts, u_ic

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes nonlinear Klein-Gordon residual:
            R = u_tt - c^2 * (u_xx + u_yy) + alpha * u + beta * u^3 - f(x, y, t)
        """
        x_interior.requires_grad_(True)
        u = model(x_interior)

        grad_u = torch.autograd.grad(u.sum(), x_interior, create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        u_xx = torch.autograd.grad(u_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        u_tt = torch.autograd.grad(u_t.sum(), x_interior, create_graph=True)[0][:, 2:3]

        lap_u = u_xx + u_yy
        f = self._forcing(x_interior)

        res = u_tt - (self.c**2) * lap_u + self.alpha * u + self.beta * (u**3) - f
        return res

    def compute_initial_loss(
        self, model: nn.Module, x_ic: torch.Tensor, u_ic_exact: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates initial value error u(x, y, 0) and initial velocity error u_t(x, y, 0) == 0."""
        x_ic.requires_grad_(True)
        u_pred = model(x_ic)
        loss_u0 = torch.mean((u_pred - u_ic_exact) ** 2)

        grad_u = torch.autograd.grad(u_pred.sum(), x_ic, create_graph=True)[0]
        u_t = grad_u[:, 2:3]
        loss_ut0 = torch.mean(u_t ** 2)

        return loss_u0 + loss_ut0
