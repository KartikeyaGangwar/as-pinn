import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE


class QuantumNonlinearSchrodinger2D(BasePDE):
    """
    2D Time-Dependent Non-Linear Schrödinger (NLS) Equation (Gross-Pitaevskii / BEC Soliton):
        i * psi_t + 0.5 * (psi_xx + psi_yy) + |psi|^2 * psi = f(x, y, t)
        
    on (x, y, t) in [-3, 3]^2 x [0, pi/2].
    
    Coupled Real-Imaginary Formulation:
        psi(x, y, t) = u(x, y, t) + i * v(x, y, t)
        
        R_u = u_t + 0.5 * (v_xx + v_yy) + (u^2 + v^2) * v - f_u = 0
        R_v = -v_t + 0.5 * (u_xx + u_yy) + (u^2 + v^2) * u - f_v = 0
        
    Exact Analytical Quantum Bright Soliton with Phase Rotation:
        psi*(x, y, t) = 2 * sech(x) * exp(i * (2*y + 1.5*t))
        u*(x, y, t) = 2 * sech(x) * cos(2*y + 1.5*t)
        v*(x, y, t) = 2 * sech(x) * sin(2*y + 1.5*t)
        |psi*|^2 = 4 * sech^2(x)
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=3, out_dim=2, device=device)
        self.bounds = ((-3.0, 3.0), (-3.0, 3.0), (0.0, np.pi / 2.0))

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns exact probability density |psi|^2 = u^2 + v^2.
        Returns [|psi|^2] with shape (N, 1) for standardized field comparison.
        """
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        t_c = x[:, 2:3]

        sech_x = 1.0 / torch.cosh(x_c)
        phase = 2.0 * y_c + 1.5 * t_c

        u = 2.0 * sech_x * torch.cos(phase)
        v = 2.0 * sech_x * torch.sin(phase)
        prob_density = u**2 + v**2
        return prob_density

    def exact_components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns exact [u, v] components."""
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        t_c = x[:, 2:3]

        sech_x = 1.0 / torch.cosh(x_c)
        phase = 2.0 * y_c + 1.5 * t_c

        u = 2.0 * sech_x * torch.cos(phase)
        v = 2.0 * sech_x * torch.sin(phase)
        return u, v

    def _forcing(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates exact analytical forcing [f_u, f_v]."""
        u_star, v_star = self.exact_components(x)
        mod_sq = u_star**2 + v_star**2
        f_u = -3.0 * v_star + 0.75 * mod_sq * v_star
        f_v = -3.0 * u_star + 0.75 * mod_sq * u_star
        return f_u, f_v

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Sample interior collocation points in (-3, 3)^2 x (0, pi/2)."""
        xy = torch.empty(n_samples, 2, device=self.device).uniform_(-3.0, 3.0)
        t = torch.empty(n_samples, 1, device=self.device).uniform_(0.0, float(np.pi / 2.0))
        return torch.cat([xy, t], dim=1)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample spatial boundary points on dOmega x [0, pi/2]."""
        n_per_wall = n_samples // 4
        t = torch.empty(n_per_wall, 1, device=self.device).uniform_(0.0, float(np.pi / 2.0))
        s = torch.empty(n_per_wall, 1, device=self.device).uniform_(-3.0, 3.0)

        b1 = torch.cat([s, torch.full_like(s, -3.0), t], dim=1)
        b2 = torch.cat([s, torch.full_like(s, 3.0), t], dim=1)
        b3 = torch.cat([torch.full_like(s, -3.0), s, t], dim=1)
        b4 = torch.cat([torch.full_like(s, 3.0), s, t], dim=1)

        pts = torch.cat([b1, b2, b3, b4], dim=0)
        u_bc, v_bc = self.exact_components(pts)
        targets = torch.cat([u_bc, v_bc], dim=1)
        return pts, targets

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample t = 0 initial condition points."""
        xy = torch.empty(n_samples, 2, device=self.device).uniform_(-3.0, 3.0)
        t0 = torch.zeros(n_samples, 1, device=self.device)
        pts = torch.cat([xy, t0], dim=1)
        u_ic, v_ic = self.exact_components(pts)
        targets = torch.cat([u_ic, v_ic], dim=1)
        return pts, targets

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes coupled real-imaginary NLS residuals:
            R_u = u_t + 0.5 * (v_xx + v_yy) + (u^2 + v^2) * v - f_u = 0
            R_v = -v_t + 0.5 * (u_xx + u_yy) + (u^2 + v^2) * u - f_v = 0
        """
        x_interior.requires_grad_(True)
        out = model(x_interior)
        u = out[:, 0:1]
        v = out[:, 1:2]

        grad_u = torch.autograd.grad(u.sum(), x_interior, create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        grad_v = torch.autograd.grad(v.sum(), x_interior, create_graph=True)[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        v_t = grad_v[:, 2:3]

        u_xx = torch.autograd.grad(u_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        lap_u = u_xx + u_yy

        v_xx = torch.autograd.grad(v_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        lap_v = v_xx + v_yy

        mod_sq = u**2 + v**2
        f_u, f_v = self._forcing(x_interior)

        res_u = u_t + 0.5 * lap_v + mod_sq * v - f_u
        res_v = -v_t + 0.5 * lap_u + mod_sq * u - f_v

        return torch.cat([res_u, res_v], dim=1)

    def compute_boundary_loss(
        self, model: nn.Module, x_bc: torch.Tensor, u_bc_exact: torch.Tensor
    ) -> torch.Tensor:
        pred = model(x_bc)
        return torch.mean((pred - u_bc_exact) ** 2)

    def compute_relative_l2_error(
        self, model: nn.Module, x_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        if x_test is None:
            x_test = self.sample_interior(n_test)
        with torch.no_grad():
            out = model(x_test)
            u_pred = out[:, 0:1]
            v_pred = out[:, 1:2]
            u_exact, v_exact = self.exact_components(x_test)
            
            err_sq = (u_pred - u_exact)**2 + (v_pred - v_exact)**2
            ref_sq = u_exact**2 + v_exact**2
            
            l2_err = torch.sqrt(torch.sum(err_sq))
            l2_ref = torch.sqrt(torch.sum(ref_sq)).clamp_min(1e-8)
            rel_error = (l2_err / l2_ref).item()
        return rel_error

    def compute_linf_error(
        self, model: nn.Module, x_test: Optional[torch.Tensor] = None, n_test: int = 2000
    ) -> float:
        if x_test is None:
            x_test = self.sample_interior(n_test)
        with torch.no_grad():
            out = model(x_test)
            u_pred = out[:, 0:1]
            v_pred = out[:, 1:2]
            u_exact, v_exact = self.exact_components(x_test)
            
            err = torch.sqrt((u_pred - u_exact)**2 + (v_pred - v_exact)**2)
            max_err = torch.max(err).item()
        return max_err
