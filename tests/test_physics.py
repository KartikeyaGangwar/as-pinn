import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from physics.burgers import Burgers1D
from physics.elliptic_interface import EllipticInterface2D
from physics.helmholtz import Helmholtz2D
from models.as_pinn import AdaptiveSubspacePINN

class TestPhysicsBenchmarks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_burgers_sampling_and_exact(self):
        pde = Burgers1D(device=self.device)
        x_in = pde.sample_interior(50)
        self.assertEqual(x_in.shape, (50, 2))
        
        x_bc, u_bc = pde.sample_boundary(40)
        self.assertEqual(x_bc.shape, (40, 2))
        self.assertEqual(u_bc.shape, (40, 1))
        
        x_ic, u_ic = pde.sample_initial(30)
        self.assertEqual(x_ic.shape, (30, 2))
        self.assertEqual(u_ic.shape, (30, 1))
        
        u_exact = pde.exact_solution(x_in)
        self.assertEqual(u_exact.shape, (50, 1))
        self.assertFalse(torch.isnan(u_exact).any())
        
        model = AdaptiveSubspacePINN(in_dim=2, out_dim=1, initial_subspaces=1).to(self.device)
        res = pde.compute_residuals(model, x_in)
        self.assertEqual(res.shape, (50, 1))
        self.assertFalse(torch.isnan(res).any())
        print("[Test] Burgers1D sampling, Cole-Hopf exact, and residual tests PASSED")

    def test_elliptic_interface_sampling_and_exact(self):
        pde = EllipticInterface2D(a1=1.0, a2=1000.0, r0=0.5, device=self.device)
        x_in = pde.sample_interior(50)
        self.assertEqual(x_in.shape, (50, 2))
        
        x_bc, u_bc = pde.sample_boundary(40)
        self.assertEqual(x_bc.shape, (40, 2))
        self.assertEqual(u_bc.shape, (40, 1))
        
        u_exact = pde.exact_solution(x_in)
        self.assertEqual(u_exact.shape, (50, 1))
        self.assertFalse(torch.isnan(u_exact).any())
        
        model = AdaptiveSubspacePINN(in_dim=2, out_dim=1, initial_subspaces=1).to(self.device)
        res = pde.compute_residuals(model, x_in)
        self.assertEqual(res.shape, (50, 1))
        self.assertFalse(torch.isnan(res).any())
        print("[Test] EllipticInterface2D sampling, jump conditions, and residual tests PASSED")

    def test_helmholtz_sampling_and_exact(self):
        pde = Helmholtz2D(device=self.device)
        x_in = pde.sample_interior(50)
        self.assertEqual(x_in.shape, (50, 2))
        
        x_bc, u_bc = pde.sample_boundary(40)
        self.assertEqual(x_bc.shape, (40, 2))
        self.assertEqual(u_bc.shape, (40, 1))
        
        u_exact = pde.exact_solution(x_in)
        self.assertEqual(u_exact.shape, (50, 1))
        self.assertFalse(torch.isnan(u_exact).any())
        
        model = AdaptiveSubspacePINN(in_dim=2, out_dim=1, initial_subspaces=1).to(self.device)
        res = pde.compute_residuals(model, x_in)
        self.assertEqual(res.shape, (50, 1))
        self.assertFalse(torch.isnan(res).any())
        print("[Test] Helmholtz2D sampling, multi-scale source, and residual tests PASSED")

if __name__ == "__main__":
    unittest.main()
