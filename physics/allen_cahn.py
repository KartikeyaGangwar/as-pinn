import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE

class AllenCahn1D(BasePDE):
    """
    1D Allen-Cahn Reaction-Diffusion Equation:
        u_t - epsilon^2 * u_xx + 5 * (u^3 - u) = f(x, t),  (x, t) in [-1, 1] x [0, 1]
        
    Models phase separation, interface motion, and metastable reaction-diffusion dynamics.
    With small epsilon (e.g. 0.02), the interface layer of width O(epsilon) separates 
    stable phases u = +1 and u = -1.
    
    Exact Analytical Moving Phase Kink Solution:
        u_exact(x, t) = tanh( (x - v * t) / (epsilon * sqrt(2)) )
        
    Source Term f(x, t):
        Analytically computed from u_t - epsilon^2 * u_xx + 5 * (u^3 - u):
        f(x, t) = - (v / (epsilon * sqrt(2))) * (1 - u^2) + 4 * (u^3 - u)
        (When v = 0 and reaction factor = 1, f = 0. For general v and factor 5, f is exact).
    """
    def __init__(
        self,
        epsilon: float = 0.02,
        velocity: float = 0.5,
        reaction_coeff: float = 5.0,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=1, device=device) # input: (x, t)
        self.epsilon = epsilon
        self.v = velocity
        self.reaction_coeff = reaction_coeff
        self.x_min, self.x_max = x_range
        self.t_min, self.t_max = t_range
        self.scale = self.epsilon * np.sqrt(2.0)

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation points uniformly in (x, t) in [-1, 1] x [0, 1]."""
        x = torch.rand(n_samples, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t = torch.rand(n_samples, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        return torch.cat([x, t], dim=1)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples Dirichlet boundary points at x = -1 and x = 1 for t in [0, 1]."""
        n_half = n_samples // 2
        t_left = torch.rand(n_half, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        x_left = torch.full((n_half, 1), self.x_min, device=self.device)
        pts_left = torch.cat([x_left, t_left], dim=1)
        
        n_right = n_samples - n_half
        t_right = torch.rand(n_right, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        x_right = torch.full((n_right, 1), self.x_max, device=self.device)
        pts_right = torch.cat([x_right, t_right], dim=1)
        
        x_bc = torch.cat([pts_left, pts_right], dim=0)
        u_bc = self.exact_solution(x_bc)
        return x_bc, u_bc

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples initial condition points at t = 0."""
        x = torch.rand(n_samples, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t = torch.zeros(n_samples, 1, device=self.device)
        x_ic = torch.cat([x, t], dim=1)
        u_ic = self.exact_solution(x_ic)
        return x_ic, u_ic

    def exact_solution(self, xt: torch.Tensor) -> torch.Tensor:
        """Exact analytical moving phase boundary kink solution."""
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = (x - self.v * t) / self.scale
        return torch.tanh(z)

    def source_term(self, xt: torch.Tensor) -> torch.Tensor:
        """Exact analytical forcing function."""
        u = self.exact_solution(xt)
        # u_t = - (v / scale) * (1 - u^2)
        # - epsilon^2 * u_xx = - (u^3 - u)
        # reaction = 5 * (u^3 - u)
        # sum = - (v / scale) * (1 - u^2) + 4 * (u^3 - u)
        u_t = -(self.v / self.scale) * (1.0 - u ** 2)
        reaction_diff = (self.reaction_coeff - 1.0) * (u ** 3 - u)
        return u_t + reaction_diff

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes PDE residual:
            r(x, t) = u_t - epsilon^2 * u_xx + reaction_coeff * (u^3 - u) - f(x, t)
        """
        x_in = x_interior.clone().detach().requires_grad_(True)
        u = model(x_in) # [N, 1]
        
        # 1st-order gradients: [u_x, u_t]
        grads = torch.autograd.grad(
            u, x_in,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        
        # 2nd-order spatial derivative: u_xx
        u_xx = torch.autograd.grad(
            u_x, x_in,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        # Reaction term
        reaction = self.reaction_coeff * (u ** 3 - u)
        f = self.source_term(x_in)
        
        residual = u_t - (self.epsilon ** 2) * u_xx + reaction - f
        return residual
