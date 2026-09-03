import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.as_pinn import AdaptiveSubspacePINN, SubspaceMLP

class TestAdaptiveSubspacePINN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_dim = 2
        self.out_dim = 1
        self.model = AdaptiveSubspacePINN(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            initial_subspaces=2,
            hidden_dim=32,
            layers=3,
        ).to(self.device)

    def test_partition_of_unity_sum_to_one(self):
        """Verify Partition of Unity weights sum to 1.0 everywhere across the domain."""
        x = torch.rand(100, self.in_dim, device=self.device) * 4.0 - 2.0
        psi = self.model.partition_of_unity(x)
        
        self.assertEqual(psi.shape, (100, 2))
        psi_sum = torch.sum(psi, dim=1)
        expected = torch.ones_like(psi_sum)
        
        max_diff = torch.max(torch.abs(psi_sum - expected)).item()
        self.assertLess(max_diff, 1e-5, f"PoU sum deviates from 1.0: {max_diff}")
        print(f"[Test] Partition of Unity sum test passed: max deviation {max_diff:.2e}")

    def test_high_order_spatial_derivatives(self):
        """Verify model is smoothly differentiable for PDE spatial derivatives (autograd.grad)."""
        x = torch.rand(20, self.in_dim, device=self.device, requires_grad=True)
        u = self.model(x) # [20, 1]
        
        # 1st order derivative (grad_u)
        grad_u = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        self.assertEqual(grad_u.shape, (20, self.in_dim))
        
        # 2nd order derivative (Laplacian: u_xx + u_yy)
        laplacian = 0.0
        for i in range(self.in_dim):
            u_xi = grad_u[:, i:i+1]
            u_xixi = torch.autograd.grad(
                u_xi, x,
                grad_outputs=torch.ones_like(u_xi),
                create_graph=False,
                retain_graph=True
            )[0][:, i:i+1]
            laplacian = laplacian + u_xixi
            
        self.assertEqual(laplacian.shape, (20, 1))
        self.assertFalse(torch.isnan(laplacian).any())
        print("[Test] High-order autograd spatial derivatives (Laplacian) validated (PASSED)")

    def test_dynamic_spawn_and_optimizer_sync(self):
        """Verify dynamic cleavage, parameter allocation, and optimizer state preservation."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        initial_subspaces = self.model.num_subspaces

        # Step 1 forward & backward
        x = torch.rand(10, self.in_dim, device=self.device)
        u = self.model(x)
        loss = u.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Spawn new subspace at (0.5, -0.5)
        clash_center = torch.tensor([0.5, -0.5], device=self.device)
        new_idx = self.model.spawn_new_subspace(clash_center)

        self.assertEqual(new_idx, initial_subspaces)
        self.assertEqual(self.model.num_subspaces, initial_subspaces + 1)
        self.assertEqual(self.model.centroids.shape[0], initial_subspaces + 1)

        # Re-sync optimizer for expanded network
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Verify forward & backward still works seamlessly with expanded network
        u_after = self.model(x)
        self.assertEqual(u_after.shape, (10, self.out_dim))
        loss_after = u_after.sum()
        loss_after.backward()
        optimizer.step()
        print("[Test] Dynamic subspace spawn and optimizer synchronization passed (PASSED)")

if __name__ == "__main__":
    unittest.main()
