import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from physics.base_pde import BasePDE

class CShapeElectromagnet(BasePDE):
    """
    2D C-Shape Electromagnet Inverse Design Benchmark (from Paper 1 / RPS_2026):
        - Poisson / Magnetostatics PDE: - div( grad(A_z) ) = mu_0 * J_z(x, y)
        - High-Permeability Iron Core Interface: dA_z / dn = 0 along iron boundary Gamma_if
        - Inverse Query Design: B_y = - dA_z / dx = B_target = -0.55 T in central air gap Omega_query
        - Far-Field Boundary: A_z = 0 at r = R_outer = 8.0
        
    Creates severe multi-objective gradient conflict between the stiff iron interface (L_if)
    and the interior uniform field requirement (L_des).
    """
    MU_IRON = 1000.0
    MU_VAC = 1.0
    J_COIL1 = 0.5
    J_COIL2 = -0.5
    B_TARGET = -0.55
    R_OUTER = 8.0

    IRON_BOXES = [
        [-3.0, -2.0, -3.0, 3.0],   # Left vertical back iron
        [-2.0, 2.0, 2.0, 3.0],     # Top horizontal iron pole
        [-2.0, 2.0, -3.0, -2.0]    # Bottom horizontal iron pole
    ]
    COIL1_BOX = [2.0, 3.0, 2.0, 3.0]     # Top coil (J = +0.5)
    COIL2_BOX = [2.0, 3.0, -3.0, -2.0]   # Bottom coil (J = -0.5)
    QUERY_BOX = [-1.5, 1.5, -1.0, 1.0]   # Central gap design zone

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(in_dim=2, out_dim=1, device=device)

    @classmethod
    def in_box(cls, x: torch.Tensor, y: torch.Tensor, box: List[float]) -> torch.Tensor:
        return (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])

    @classmethod
    def in_iron(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = torch.zeros_like(x, dtype=torch.bool)
        for b in cls.IRON_BOXES:
            m = m | cls.in_box(x, y, b)
        return m

    @classmethod
    def in_coils(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cls.in_box(x, y, cls.COIL1_BOX) | cls.in_box(x, y, cls.COIL2_BOX)

    def sample_interior(self, n_samples: int) -> torch.Tensor:
        """Sample free space air points in domain (excluding iron and coils)."""
        pts_list = []
        accumulated = 0
        while accumulated < n_samples:
            cand = torch.empty(n_samples * 2, 2, device=self.device).uniform_(-self.R_OUTER, self.R_OUTER)
            r = torch.norm(cand, dim=1)
            inside = (r < self.R_OUTER) & (~self.in_iron(cand[:, 0], cand[:, 1])) & (~self.in_coils(cand[:, 0], cand[:, 1]))
            valid = cand[inside]
            pts_list.append(valid)
            accumulated += valid.shape[0]
        return torch.cat(pts_list, dim=0)[:n_samples]

    def sample_coils(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample excitation current coil points."""
        n_half = n_samples // 2
        c1 = torch.empty(n_half, 2, device=self.device)
        c1[:, 0].uniform_(self.COIL1_BOX[0], self.COIL1_BOX[1])
        c1[:, 1].uniform_(self.COIL1_BOX[2], self.COIL1_BOX[3])
        j1 = torch.full((n_half, 1), self.J_COIL1, device=self.device)

        c2 = torch.empty(n_half, 2, device=self.device)
        c2[:, 0].uniform_(self.COIL2_BOX[0], self.COIL2_BOX[1])
        c2[:, 1].uniform_(self.COIL2_BOX[2], self.COIL2_BOX[3])
        j2 = torch.full((n_half, 1), self.J_COIL2, device=self.device)

        pts = torch.cat([c1, c2], dim=0)
        currents = torch.cat([j1, j2], dim=0)
        return pts, currents

    def sample_boundary(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample Dirichlet far-field boundary points on r = R_OUTER."""
        theta = torch.rand(n_samples, 1, device=self.device) * 2.0 * np.pi
        x = self.R_OUTER * torch.cos(theta)
        y = self.R_OUTER * torch.sin(theta)
        pts = torch.cat([x, y], dim=1)
        u_bc = torch.zeros(n_samples, 1, device=self.device)
        return pts, u_bc

    def sample_interface(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample iron core outer boundary interface points with outward normal vectors."""
        segments = [
            ([-3.0, 3.0], [-3.0, -3.0], [0.0, -1.0]),  # Bottom iron outer
            ([-3.0, 3.0], [3.0, 3.0], [0.0, 1.0]),    # Top iron outer
            ([-3.0, -3.0], [-3.0, 3.0], [-1.0, 0.0]), # Left iron outer
            ([2.0, 2.0], [2.0, 3.0], [1.0, 0.0]),     # Top pole right
            ([2.0, 2.0], [-3.0, -2.0], [1.0, 0.0]),   # Bottom pole right
            ([-2.0, 2.0], [2.0, 2.0], [0.0, -1.0]),   # Top pole inner gap face
            ([-2.0, 2.0], [-2.0, -2.0], [0.0, 1.0]),  # Bottom pole inner gap face
            ([-2.0, -2.0], [-2.0, 2.0], [1.0, 0.0]),  # Back iron inner face
        ]
        seg_idx = torch.randint(0, len(segments), (n_samples,), device=self.device)
        t = torch.rand(n_samples, 1, device=self.device)
        pts = torch.zeros(n_samples, 2, device=self.device)
        normals = torch.zeros(n_samples, 2, device=self.device)
        for idx, (x_rng, y_rng, norm) in enumerate(segments):
            mask = (seg_idx == idx)
            count = mask.sum().item()
            if count > 0:
                t_sub = t[mask]
                px = x_rng[0] + t_sub * (x_rng[1] - x_rng[0])
                py = y_rng[0] + t_sub * (y_rng[1] - y_rng[0])
                pts[mask] = torch.cat([px, py], dim=1)
                normals[mask] = torch.tensor(norm, device=self.device, dtype=pts.dtype).repeat(count, 1)
        return pts, normals

    def sample_query(self, n_samples: int) -> torch.Tensor:
        """Sample query points inside the central air gap design zone."""
        q = torch.empty(n_samples, 2, device=self.device)
        q[:, 0].uniform_(self.QUERY_BOX[0], self.QUERY_BOX[1])
        q[:, 1].uniform_(self.QUERY_BOX[2], self.QUERY_BOX[3])
        return q

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Ideal target vector potential field A_z = -B_target * x generating uniform B_y = B_target."""
        return -self.B_TARGET * x[:, 0:1]

    def compute_residuals(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Evaluates magnetostatic PDE residual -Laplacian(A_z) = 0 in air."""
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
        return -lap_A

    def compute_interface_loss(self, model: nn.Module, i_pts: torch.Tensor, i_normals: torch.Tensor) -> torch.Tensor:
        """Computes Neumann interface penalty: dA_z / dn = 0 along iron boundary."""
        i_req = i_pts.clone().detach().requires_grad_(True)
        A = model(i_req)
        grad_A = torch.autograd.grad(
            A, i_req,
            grad_outputs=torch.ones_like(A),
            create_graph=True,
            retain_graph=True
        )[0]
        dA_dn = (grad_A * i_normals).sum(dim=1, keepdim=True)
        return torch.mean(dA_dn ** 2)

    def compute_design_loss(self, model: nn.Module, q_pts: torch.Tensor) -> torch.Tensor:
        """Computes inverse design target loss: B_y = -dA_z/dx == B_TARGET."""
        q_req = q_pts.clone().detach().requires_grad_(True)
        A = model(q_req)
        grad_A = torch.autograd.grad(
            A, q_req,
            grad_outputs=torch.ones_like(A),
            create_graph=True,
            retain_graph=True
        )[0]
        By = -grad_A[:, 0:1]
        return torch.mean((By - self.B_TARGET) ** 2)

    def compute_design_rmse(self, model: nn.Module, n_query: int = 1000) -> float:
        """Computes root-mean-squared error of By relative to B_TARGET in air gap."""
        q_pts = self.sample_query(n_query)
        q_req = q_pts.clone().detach().requires_grad_(True)
        A = model(q_req)
        grad_A = torch.autograd.grad(A, q_req, grad_outputs=torch.ones_like(A))[0]
        By = -grad_A[:, 0:1]
        rmse = torch.sqrt(torch.mean((By - self.B_TARGET) ** 2)).item()
        return rmse

    def compute_relative_l2_error(self, model: nn.Module, n_test: int = 2000) -> float:
        """For C-Shape benchmark, returns Design By RMSE as the target accuracy metric."""
        return self.compute_design_rmse(model, n_query=n_test)

    def compute_linf_error(self, model: nn.Module, n_test: int = 2000) -> float:
        """Max absolute design error |By - B_target|."""
        q_pts = self.sample_query(n_test)
        q_req = q_pts.clone().detach().requires_grad_(True)
        A = model(q_req)
        grad_A = torch.autograd.grad(A, q_req, grad_outputs=torch.ones_like(A))[0]
        By = -grad_A[:, 0:1]
        linf = torch.max(torch.abs(By - self.B_TARGET)).item()
        return linf
