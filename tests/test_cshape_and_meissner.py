import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
from physics.cshape_magnet import CShapeElectromagnet
from physics.meissner_cylinder import MeissnerCylinder
from benchmarks.models_baseline import CAGradOptimizer, PCGradOptimizer, MonolithicMLP

def test_cshape_electromagnet_physics():
    pde = CShapeElectromagnet(device=torch.device("cpu"))
    
    # 1. Test sampling
    x_int = pde.sample_interior(200)
    assert x_int.shape == (200, 2)
    assert not pde.in_iron(x_int[:, 0], x_int[:, 1]).any()
    assert not pde.in_coils(x_int[:, 0], x_int[:, 1]).any()
    
    x_coils, j_coils = pde.sample_coils(100)
    assert x_coils.shape == (100, 2)
    assert j_coils.shape == (100, 1)
    
    x_bc, u_bc = pde.sample_boundary(100)
    assert x_bc.shape == (100, 2)
    assert u_bc.shape == (100, 1)
    
    x_if, n_if = pde.sample_interface(100)
    assert x_if.shape == (100, 2)
    assert n_if.shape == (100, 2)
    
    q_pts = pde.sample_query(100)
    assert q_pts.shape == (100, 2)
    
    # 2. Test residual autograd with dummy model
    model = MonolithicMLP(in_dim=2, out_dim=1, hidden_dim=32, layers=3)
    res = pde.compute_residuals(model, x_int[:50])
    assert res.shape == (50, 1)
    
    loss_if = pde.compute_interface_loss(model, x_if[:50], n_if[:50])
    assert loss_if.item() >= 0.0
    
    loss_des = pde.compute_design_loss(model, q_pts[:50])
    assert loss_des.item() >= 0.0
    
    rmse = pde.compute_design_rmse(model, n_query=100)
    assert rmse > 0.0
    print(f"\n[+] CShapeElectromagnet verified! Init Design RMSE: {rmse:.4f}")

def test_meissner_cylinder_analytical():
    pde = MeissnerCylinder(r0=1.0, r_outer=4.0, b0=1.0, device=torch.device("cpu"))
    
    # Test interior zero screening
    x_inside = torch.tensor([[0.2, 0.3], [0.0, 0.0], [-0.5, 0.5]], dtype=torch.float32)
    u_inside = pde.exact_solution(x_inside)
    assert torch.all(u_inside == 0.0)
    
    # Test exterior Laplace satisfaction
    x_ext = pde.sample_interior(100).requires_grad_(True)
    
    # Evaluate Laplace on exact analytical solution
    class ExactWrapper(nn.Module):
        def forward(self, x):
            return pde.exact_solution(x)
            
    res = pde.compute_residuals(ExactWrapper(), x_ext)
    max_res = torch.max(torch.abs(res)).item()
    assert max_res < 1e-4, f"Analytical solution should satisfy Laplace: max res = {max_res}"
    print(f"\n[+] Meissner Cylinder verified! Analytical Laplace Residual: {max_res:.2e}")

def test_cagrad_optimizer():
    model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
    raw_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cagrad = CAGradOptimizer(raw_opt, c=0.4)
    
    x = torch.randn(20, 2)
    y1 = model(x)
    y2 = model(x * 2.0)
    
    loss1 = torch.mean(y1 ** 2)
    loss2 = torch.mean((y2 - 1.0) ** 2)
    
    cagrad.step([loss1, loss2], model)
    print("\n[+] CAGrad step verified on multi-objective losses!")

if __name__ == "__main__":
    test_cshape_electromagnet_physics()
    test_meissner_cylinder_analytical()
    test_cagrad_optimizer()
