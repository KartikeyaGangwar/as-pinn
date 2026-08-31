import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE

class Helmholtz2D(BasePDE):
    """
    2D High-Frequency Helmholtz Equation:
        Delta u(x, y) + k^2 * u(x, y) = f(x, y),   (x, y) in [-1, 1]^2
        
    With high wavenumber k = 4*pi, 8*pi, or 16*pi, inducing severe multi-scale
    destructive interference and spectral bias in standard global MLPs.
    
    Exact manufactured solution:
        u(x, y) = sin(a1 * pi * x) * sin(a2 * pi * y) + alpha * sin(a3 * pi * x) * sin(a4 * pi * y)
    Source term:
        f(x, y) = (k^2 - pi^2*(a1^2 + a2^2)) * sin(a1*pi*x)*sin(a2*pi*y) 
                + alpha * (k^2 - pi^2*(a3^2 + a4^2)) * sin(a3*pi*x)*sin(a4*pi*y)
    """
    def __init__(
        self,
        k: float = 4.0 * np.pi,
        modes: Tuple[float, float, float, float] = (2.0, 2.0, 4.0, 4.0),
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=1, device=device)
        self.k = k
        self.a1, self.a2, self.a3, self.a4 = modes
        self.alpha = alpha

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation points uniformly in [-1, 1]^2."""
        return torch.rand(n_samples, 2, device=self.device) * 2.0 - 1.0

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples Dirichlet boundary points on [-1, 1]^2."""
        n_side = n_samples // 4
        s = torch.rand(n_side, 1, device=self.device) * 2.0 - 1.0
        
        bottom = torch.cat([s, torch.full_like(s, -1.0)], dim=1)
        top = torch.cat([s, torch.full_like(s, 1.0)], dim=1)
        left = torch.cat([torch.full_like(s, -1.0), s], dim=1)
        right = torch.cat([torch.full_like(s, 1.0), s], dim=1)
        
        x_bc = torch.cat([bottom, top, left, right], dim=0)
        u_bc = self.exact_solution(x_bc)
        return x_bc, u_bc

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates analytic source term f(x, y)."""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        
        term1_coeff = (self.k ** 2) - (np.pi ** 2) * (self.a1 ** 2 + self.a2 ** 2)
        term1 = term1_coeff * torch.sin(self.a1 * np.pi * x_coord) * torch.sin(self.a2 * np.pi * y_coord)
        
        term2_coeff = (self.k ** 2) - (np.pi ** 2) * (self.a3 ** 2 + self.a4 ** 2)
        term2 = self.alpha * term2_coeff * torch.sin(self.a3 * np.pi * x_coord) * torch.sin(self.a4 * np.pi * y_coord)
        
        return term1 + term2

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes Helmholtz residual:
            r(x, y) = Delta u + k^2 * u - f(x, y) = u_xx + u_yy + k^2 * u - f(x, y)
        """
        x_in = x_interior.clone().detach().requires_grad_(True)
        u = model(x_in) # [N, 1]
        
        grads = torch.autograd.grad(
            u, x_in,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        
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
        
        laplacian = u_xx + u_yy
        f = self.source_term(x_in)
        residual = laplacian + (self.k ** 2) * u - f
        return residual

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Exact analytical solution."""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        
        mode1 = torch.sin(self.a1 * np.pi * x_coord) * torch.sin(self.a2 * np.pi * y_coord)
        mode2 = self.alpha * torch.sin(self.a3 * np.pi * x_coord) * torch.sin(self.a4 * np.pi * y_coord)
        return mode1 + mode2
