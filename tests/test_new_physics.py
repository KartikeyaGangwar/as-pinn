import unittest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from physics.allen_cahn import AllenCahn1D
from physics.high_dim_diffusion import HighDimDiffusion4D

class ExactSolutionWrapper(torch.nn.Module):
    def __init__(self, exact_fn):
        super().__init__()
        self.exact_fn = exact_fn
        
    def forward(self, x):
        return self.exact_fn(x)

class TestNewPhysicsSuite(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_allen_cahn_exact_and_residual(self):
        pde = AllenCahn1D(epsilon=0.02, velocity=0.5, reaction_coeff=5.0, device=self.device)
        
        # Test sampling
        x_int = pde.sample_interior(100)
        self.assertEqual(x_int.shape, (100, 2))
        
        x_bc, u_bc = pde.sample_boundary(64)
        self.assertEqual(x_bc.shape, (64, 2))
        self.assertEqual(u_bc.shape, (64, 1))
        
        x_ic, u_ic = pde.sample_initial(64)
        self.assertEqual(x_ic.shape, (64, 2))
        self.assertEqual(u_ic.shape, (64, 1))
        
        # Test exact solution differential residual: r(x, t) must be identically 0
        wrapper = ExactSolutionWrapper(pde.exact_solution)
        res = pde.compute_residuals(wrapper, x_int)
        max_res = torch.max(torch.abs(res)).item()
        print(f"\n[Test] Allen-Cahn 1D Exact Solution Max Residual: {max_res:.6e}")
        self.assertLess(max_res, 1e-4)
        
    def test_high_dim_diffusion_exact_and_residual(self):
        pde = HighDimDiffusion4D(kappa=0.1, alpha=0.5, device=self.device)
        
        # Test sampling (5D tensor: x1, x2, x3, x4, t)
        x_int = pde.sample_interior(100)
        self.assertEqual(x_int.shape, (100, 5))
        
        x_bc, u_bc = pde.sample_boundary(64)
        self.assertEqual(x_bc.shape, (64, 5))
        self.assertEqual(u_bc.shape, (64, 1))
        
        x_ic, u_ic = pde.sample_initial(64)
        self.assertEqual(x_ic.shape, (64, 5))
        self.assertEqual(u_ic.shape, (64, 1))
        
        # Test exact solution differential residual: r(x, t) must be identically 0
        wrapper = ExactSolutionWrapper(pde.exact_solution)
        res = pde.compute_residuals(wrapper, x_int)
        max_res = torch.max(torch.abs(res)).item()
        print(f"[Test] 4D High-Dim Diffusion Exact Solution Max Residual: {max_res:.6e}")
        self.assertLess(max_res, 1e-4)

if __name__ == "__main__":
    unittest.main()
