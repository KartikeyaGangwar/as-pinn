import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from physics.base_pde import BasePDE

class MeissnerCylinder(BasePDE):
    """
    Superconducting Meissner Cylinder (Analytical Multi-Physics Benchmark from Paper 1):
        - By the Meissner-Ochsenfeld effect, magnetic flux is completely expelled from the superconductor:
            A(x, y) = 0,                                                    r < R_0
            A(x, y) = B_0 * y * (1.0 - R_0^2 / (x^2 + y^2)),                r >= R_0
        - Governing PDE: Laplace equation div(grad(A)) = 0 in the exterior domain r > R_0.
        - Cylinder Surface Boundary: A = 0 and dA/dr = 0 at r = R_0.
        - Outer Box Boundary: A(x, y) = A_exact(x, y) on [-R_outer, R_outer]^2.
        
    Tests the network's ability to maintain exact zero interior field penetration 
    while preserving analytical dipole decay in the surrounding vacuum.
    """
    def __init__(
        self,
        r0: float = 1.0,
        r_outer: float = 4.0,
        b0: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_dim=2, out_dim=1, device=device)
        self.r0 = r0
        self.r_outer = r_outer
        self.b0 = b0

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Computes analytical magnetic vector potential field."""
        px = x[:, 0:1]
        py = x[:, 1:2]
        r_sq = px ** 2 + py ** 2
        r = torch.sqrt(torch.clamp(r_sq, min=1e-8))
        
        # Exterior solution (r >= r0) with safe division
        safe_r_sq = torch.clamp(r_sq, min=self.r0 ** 2)
        a_ext = self.b0 * py * (1.0 - (self.r0 ** 2) / safe_r_sq)
        
        # Interior solution (r < r0) is identically zero (Meissner flux exclusion)
        return torch.where(r >= self.r0, a_ext, torch.zeros_like(py))

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Samples exterior vacuum points r > r0 inside [-r_outer, r_outer]^2."""
        pts_list = []
        accumulated = 0
        while accumulated < n_samples:
            cand = torch.empty(n_samples * 2, 2, device=self.device).uniform_(-self.r_outer, self.r_outer)
            r = torch.norm(cand, dim=1)
            valid = cand[r > self.r0]
            pts_list.append(valid)
            accumulated += valid.shape[0]
        return torch.cat(pts_list, dim=0)[:n_samples]

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples outer box boundary points and cylinder inner interface points."""
        n_outer = n_samples // 2
        n_inner = n_samples - n_outer
        
        # Outer box edges [-r_outer, r_outer]^2
        n_side = n_outer // 4
        s = torch.rand(n_side, 1, device=self.device) * 2.0 * self.r_outer - self.r_outer
        b_bot = torch.cat([s, torch.full_like(s, -self.r_outer)], dim=1)
        b_top = torch.cat([s, torch.full_like(s, self.r_outer)], dim=1)
        b_left = torch.cat([torch.full_like(s, -self.r_outer), s], dim=1)
        b_right = torch.cat([torch.full_like(s, self.r_outer), s], dim=1)
        x_outer = torch.cat([b_bot, b_top, b_left, b_right], dim=0)
        u_outer = self.exact_solution(x_outer)
        
        # Inner cylinder surface r = r0 (A = 0)
        theta = torch.rand(n_inner, 1, device=self.device) * 2.0 * np.pi
        x_inner = torch.cat([self.r0 * torch.cos(theta), self.r0 * torch.sin(theta)], dim=1)
        u_inner = torch.zeros(n_inner, 1, device=self.device)
        
        x_bc = torch.cat([x_outer, x_inner], dim=0)
        u_bc = torch.cat([u_outer, u_inner], dim=0)
        return x_bc, u_bc

    def compute_residuals(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Evaluates Laplace PDE residual div(grad(A)) = 0 in exterior vacuum."""
        x_req = x.clone().detach().requires_grad_(True)
        A = model(x_req)
        
        grad_A = torch.autograd.grad(
            A, x_req,
            grad_outputs=torch.ones_like(A),
            create_graph=True,
            retain_graph=True
        )[0]
        
        A_x = grad_A[:, 0:1]
        A_y = grad_A[:, 1:2]
        
        A_xx = torch.autograd.grad(A_x, x_req, grad_outputs=torch.ones_like(A_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        A_yy = torch.autograd.grad(A_y, x_req, grad_outputs=torch.ones_like(A_y), create_graph=True, retain_graph=True)[0][:, 1:2]
        
        lap_A = A_xx + A_yy
        return lap_A

    def compute_relative_l2_error(self, model: nn.Module, n_test: int = 2000) -> float:
        """Computes relative L2 error vs exact analytical solution in exterior vacuum."""
        x_test = self.sample_interior(n_test)
        u_exact = self.exact_solution(x_test)
        with torch.no_grad():
            u_pred = model(x_test)
        l2_err = torch.norm(u_pred - u_exact) / (torch.norm(u_exact) + 1e-8)
        return float(l2_err.item())

    def compute_linf_error(self, model: nn.Module, n_test: int = 2000) -> float:
        """Computes Linf error in exterior vacuum."""
        x_test = self.sample_interior(n_test)
        u_exact = self.exact_solution(x_test)
        with torch.no_grad():
            u_pred = model(x_test)
        linf = torch.max(torch.abs(u_pred - u_exact)).item()
        return float(linf)
