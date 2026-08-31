import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE

class EllipticInterface2D(BasePDE):
    """
    2D High-Contrast Elliptic Interface Problem:
        - div( a(x, y) * grad(u(x, y)) ) = f(x, y),   (x, y) in [-1, 1]^2
        
    Domain contains a circular inclusion of radius r_0 = 0.5:
        a(x, y) = a_1 (interior: r < r_0)
        a(x, y) = a_2 (exterior: r >= r_0)
        
    With contrast ratio a_2 / a_1 = 1000, creating a sharp gradient discontinuity across the interface.
    
    Analytical piecewise solution satisfying interface jump conditions [[u]] = 0 and [[a * du/dn]] = 0:
        u_1(x, y) = (x^2 + y^2) / a_1,                     r < r_0
        u_2(x, y) = (x^2 + y^2 - r_0^2) / a_2 + r_0^2/a_1, r >= r_0
        f(x, y) = -4.0 everywhere.
    """
    def __init__(
        self,
        a1: float = 1.0,
        a2: float = 1000.0,
        r0: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=1, device=device)
        self.a1 = a1
        self.a2 = a2
        self.r0 = r0
        self.f_const = -4.0

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation points uniformly in [-1, 1]^2."""
        return torch.rand(n_samples, 2, device=self.device) * 2.0 - 1.0

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples Dirichlet boundary points on all 4 edges of [-1, 1]^2."""
        n_side = n_samples // 4
        s = torch.rand(n_side, 1, device=self.device) * 2.0 - 1.0
        
        # Bottom: y = -1
        bottom = torch.cat([s, torch.full_like(s, -1.0)], dim=1)
        # Top: y = 1
        top = torch.cat([s, torch.full_like(s, 1.0)], dim=1)
        # Left: x = -1
        left = torch.cat([torch.full_like(s, -1.0), s], dim=1)
        # Right: x = 1
        right = torch.cat([torch.full_like(s, 1.0), s], dim=1)
        
        x_bc = torch.cat([bottom, top, left, right], dim=0)
        u_bc = self.exact_solution(x_bc)
        return x_bc, u_bc

    def get_diffusion_coefficient(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates piecewise constant diffusion coefficient a(x, y)."""
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        a = torch.where(r < self.r0, torch.tensor(self.a1, device=x.device), torch.tensor(self.a2, device=x.device))
        return a

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes PDE residual:
            r(x, y) = - div( a * grad(u) ) - f = - ( (a * u_x)_x + (a * u_y)_y ) - (-4.0)
        """
        x_in = x_interior.clone().detach().requires_grad_(True)
        u = model(x_in) # [N, 1]
        
        # First-order gradient: [u_x, u_y]
        grads = torch.autograd.grad(
            u, x_in,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        
        # Second-order derivatives of u directly
        u_xx = torch.autograd.grad(
            u_x, x_in,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            u_y, x_in,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        laplacian_u = u_xx + u_yy
        a = self.get_diffusion_coefficient(x_in)
        residual = -laplacian_u - (self.f_const / a)
        return residual

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Exact analytical solution."""
        r_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        r0_sq = self.r0 ** 2
        
        u_inside = r_sq / self.a1
        u_outside = (r_sq - r0_sq) / self.a2 + (r0_sq / self.a1)
        
        u_exact = torch.where(r_sq < r0_sq, u_inside, u_outside)
        return u_exact
