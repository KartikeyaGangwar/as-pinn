import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE

class HighDimDiffusion4D(BasePDE):
    """
    4D Space + 1D Time High-Dimensional Diffusion Equation (5D Input):
        u_t - kappa * sum_{i=1}^4 u_{x_i x_i} = 0,   (x_1..x_4) in [-1, 1]^4,  t in [0, 1]
        
    Demonstrates that AS-PINN scales to high dimensions without suffering from the 
    Curse of Dimensionality that causes grid-based methods (FEM / FBPINNs) to collapse (requiring K^4 subdomains).
    
    Exact Analytical Solution (Multi-Scale Fourier Diffusion Modes):
        u_exact(x, t) = exp(-lambda * t) * prod_{i=1}^4 sin(pi * x_i) 
                      + alpha * exp(-4 * lambda * t) * prod_{i=1}^4 cos(2 * pi * x_i)
                      
    With kappa = 0.1 and lambda = 4 * pi^2 * kappa = 0.4 * pi^2, 
    the spatial Laplacian exactly balances the time decay, yielding homogeneous PDE with f = 0.
    """
    def __init__(
        self,
        kappa: float = 0.1,
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=5, out_dim=1, device=device) # input: (x1, x2, x3, x4, t)
        self.kappa = kappa
        self.alpha = alpha
        self.decay_lambda = 4.0 * (np.pi ** 2) * self.kappa # 4 * pi^2 * kappa

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation points uniformly in [-1, 1]^4 x [0, 1]."""
        # x: [N, 4] in [-1, 1]^4
        x = torch.rand(n_samples, 4, device=self.device) * 2.0 - 1.0
        # t: [N, 1] in [0, 1]
        t = torch.rand(n_samples, 1, device=self.device)
        return torch.cat([x, t], dim=1)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples Dirichlet boundary points on all 8 faces of the 4D hypercube [-1, 1]^4.
        """
        # 8 faces (2 faces per spatial dimension i=0..3 at x_i = -1 and x_i = +1)
        samples_per_face = max(1, n_samples // 8)
        boundary_pts = []
        
        for dim in range(4):
            for val in [-1.0, 1.0]:
                x = torch.rand(samples_per_face, 4, device=self.device) * 2.0 - 1.0
                x[:, dim] = val
                t = torch.rand(samples_per_face, 1, device=self.device)
                boundary_pts.append(torch.cat([x, t], dim=1))
                
        x_bc = torch.cat(boundary_pts, dim=0)[:n_samples]
        u_bc = self.exact_solution(x_bc)
        return x_bc, u_bc

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples initial condition at t = 0 in [-1, 1]^4."""
        x = torch.rand(n_samples, 4, device=self.device) * 2.0 - 1.0
        t = torch.zeros(n_samples, 1, device=self.device)
        x_ic = torch.cat([x, t], dim=1)
        u_ic = self.exact_solution(x_ic)
        return x_ic, u_ic

    def exact_solution(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Exact analytical solution evaluating multi-scale 4D diffusion modes.
        xt shape: [N, 5] -> columns 0..3: x1..x4, column 4: t
        """
        x = xt[:, 0:4] # [N, 4]
        t = xt[:, 4:5] # [N, 1]
        
        # Mode 1: exp(-lambda * t) * prod_i sin(pi * x_i)
        sin_prod = torch.prod(torch.sin(np.pi * x), dim=1, keepdim=True) # [N, 1]
        mode1 = torch.exp(-self.decay_lambda * t) * sin_prod
        
        # Mode 2: alpha * exp(-4 * lambda * t) * prod_i cos(2 * pi * x_i)
        cos_prod = torch.prod(torch.cos(2.0 * np.pi * x), dim=1, keepdim=True) # [N, 1]
        mode2 = self.alpha * torch.exp(-4.0 * self.decay_lambda * t) * cos_prod
        
        return mode1 + mode2

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes PDE residual:
            r(x, t) = u_t - kappa * sum_{i=1}^4 u_{x_i x_i}
        """
        x_in = x_interior.clone().detach().requires_grad_(True)
        u = model(x_in) # [N, 1]
        
        # 1st-order gradients: [u_x1, u_x2, u_x3, u_x4, u_t]
        grads = torch.autograd.grad(
            u, x_in,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_t = grads[:, 4:5]
        laplacian = torch.zeros_like(u)
        
        # 2nd-order spatial derivatives: sum_{i=0}^3 u_{x_i x_i}
        for i in range(4):
            u_xi = grads[:, i:i+1]
            u_xixi = torch.autograd.grad(
                u_xi, x_in,
                grad_outputs=torch.ones_like(u_xi),
                create_graph=True,
                retain_graph=True
            )[0][:, i:i+1]
            laplacian = laplacian + u_xixi
            
        residual = u_t - self.kappa * laplacian
        return residual
