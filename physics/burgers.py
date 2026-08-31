import numpy as np
import scipy.special
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE

class Burgers1D(BasePDE):
    """
    1D Viscous Burgers' Equation:
        u_t + u * u_x - nu * u_xx = 0,   (x, t) in [-1, 1] x [0, 1]
    
    Initial Condition:
        u(x, 0) = -sin(pi * x)
    
    Boundary Conditions (Dirichlet):
        u(-1, t) = 0,  u(1, t) = 0
        
    At low viscosity (e.g. nu = 0.01 / pi = 0.003183 or nu = 0.001),
    a steep shockwave develops at x = 0 as t -> 1, triggering severe intra-network gradient conflict.
    """
    def __init__(
        self,
        nu: float = 0.01 / np.pi,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=1, device=device) # input: (x, t)
        self.nu = nu
        self.x_min, self.x_max = x_range
        self.t_min, self.t_max = t_range
        
        # Precompute Gauss-Legendre quadrature nodes for Cole-Hopf exact solver
        quad_nodes, quad_weights = np.polynomial.legendre.leggauss(256)
        self.quad_nodes = torch.tensor(quad_nodes, dtype=torch.float32, device="cpu")
        self.quad_weights = torch.tensor(quad_weights, dtype=torch.float32, device="cpu")

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples collocation points uniformly in (x, t) in (x_min, x_max) x (t_min, t_max)."""
        x = torch.rand(n_samples, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t = torch.rand(n_samples, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        return torch.cat([x, t], dim=1)

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples boundary points at x = -1 and x = 1 for t in [0, 1]."""
        n_half = n_samples // 2
        t_left = torch.rand(n_half, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        x_left = torch.full((n_half, 1), self.x_min, device=self.device)
        pts_left = torch.cat([x_left, t_left], dim=1)
        
        n_right = n_samples - n_half
        t_right = torch.rand(n_right, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        x_right = torch.full((n_right, 1), self.x_max, device=self.device)
        pts_right = torch.cat([x_right, t_right], dim=1)
        
        x_bc = torch.cat([pts_left, pts_right], dim=0)
        u_bc = torch.zeros(n_samples, 1, device=self.device)
        return x_bc, u_bc

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples initial condition points at t = 0."""
        x = torch.rand(n_samples, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t = torch.zeros(n_samples, 1, device=self.device)
        x_ic = torch.cat([x, t], dim=1)
        u_ic = -torch.sin(np.pi * x)
        return x_ic, u_ic

    def compute_residuals(self, model: nn.Module, x_interior: torch.Tensor) -> torch.Tensor:
        """
        Computes the differential residual:
            r(x, t) = u_t + u * u_x - nu * u_xx
        """
        x_in = x_interior.clone().detach().requires_grad_(True)
        u = model(x_in) # [N, 1]
        
        # 1st order gradients: [du/dx, du/dt]
        grads = torch.autograd.grad(
            u, x_in,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        
        # 2nd order derivative: d^2u / dx^2
        u_xx = torch.autograd.grad(
            u_x, x_in,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        residual = u_t + u * u_x - self.nu * u_xx
        return residual

    def exact_solution(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Exact analytical solution via vectorized Cole-Hopf transformation.
        Evaluates the true Burgers velocity field with quad integration over [-3, 3].
        """
        orig_device = xt.device
        orig_dtype = xt.dtype
        x = xt[:, 0:1].to(device=self.device, dtype=torch.float64)
        t = xt[:, 1:2].to(device=self.device, dtype=torch.float64)
        
        # Handle t = 0 explicitly: u(x, 0) = -sin(pi * x)
        t_mask = (t < 1e-6)
        u_exact = torch.zeros_like(x)
        
        if t_mask.any():
            u_exact[t_mask] = -torch.sin(np.pi * x[t_mask])
            
        non_zero_mask = (~t_mask).squeeze(-1)
        if non_zero_mask.any():
            x_nz = x[non_zero_mask] # [M, 1]
            t_nz = t[non_zero_mask] # [M, 1]
            
        # Process on CPU to use 0 GPU memory
        x_nz_cpu = x_nz.cpu()
        t_nz_cpu = t_nz.cpu()
        u_exact_cpu = u_exact.cpu()
        
        chunk_size = 256
        for start_idx in range(0, x_nz_cpu.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, x_nz_cpu.shape[0])
            x_chunk = x_nz_cpu[start_idx:end_idx]
            t_chunk = t_nz_cpu[start_idx:end_idx]
            
            eta = (self.quad_nodes * 3.0).unsqueeze(0) # [1, K]
            w = (self.quad_weights * 3.0).unsqueeze(0)  # [1, K]
            int_u0 = (1.0 / np.pi) * (torch.cos(np.pi * eta) - 1.0) # [1, K]
            diff = x_chunk - eta # [M_chunk, K]
            
            exponent = - ( (diff ** 2) / (2.0 * t_chunk) + int_u0 ) / (2.0 * self.nu)
            max_exp = torch.max(exponent, dim=1, keepdim=True).values
            exp_term = torch.exp(exponent - max_exp)
            
            denom = torch.sum(w * exp_term, dim=1, keepdim=True)
            integrand_num = (diff / t_chunk) * exp_term
            num = torch.sum(w * integrand_num, dim=1, keepdim=True)
            
            u_nz_chunk = num / denom.clamp_min(1e-15)
            u_exact_cpu[non_zero_mask.nonzero().cpu()[start_idx:end_idx, 0]] = u_nz_chunk
            
        return u_exact_cpu.to(device=orig_device, dtype=orig_dtype)
