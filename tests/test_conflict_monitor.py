import unittest
import time
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.as_pinn import AdaptiveSubspacePINN, SubspaceMLP
from models.conflict_monitor import ContinuousConflictMonitor

class TestConflictMonitor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_dim = 2
        self.out_dim = 1
        self.mlp = SubspaceMLP(self.in_dim, self.out_dim, hidden_dim=32, layers=3).to(self.device)
        self.monitor = ContinuousConflictMonitor(threshold=-0.1, ema_decay=0.8)

    def test_vmap_matches_autograd_exact(self):
        """Verify torch.func.vmap gradients match autograd per-sample gradients with float precision."""
        batch_size = 16
        x_batch = torch.randn(batch_size, self.in_dim, device=self.device)
        
        def dummy_loss(u, x):
            target = torch.sin(x[0]) + torch.cos(x[1])
            return 0.5 * (u[0] - target) ** 2

        # 1. Vectorized gradients
        G_vmap = self.monitor.compute_per_point_gradients_vmap(self.mlp, x_batch, dummy_loss)
        
        # 2. Autograd fallback gradients
        G_autograd = self.monitor.compute_per_point_gradients_autograd_fallback(self.mlp, x_batch, dummy_loss)
        
        self.assertEqual(G_vmap.shape, G_autograd.shape)
        max_diff = torch.max(torch.abs(G_vmap - G_autograd)).item()
        self.assertLess(max_diff, 1e-5, f"Vectorized gradients diverge from autograd: max diff {max_diff}")
        print(f"\n[Test] vmap vs autograd max diff: {max_diff:.2e} (PASSED)")

    def test_gram_matrix_and_conflict_detection(self):
        """Verify Gram matrix properties: diagonals = 1, symmetric, correct off-diagonal clash metrics."""
        G = torch.tensor([
            [1.0, 0.0],
            [-1.0, 0.0], # Conflicting with row 0
            [0.0, 1.0],  # Orthogonal to row 0 and 1
        ], device=self.device)
        
        C, mean_align, clash_ratio, pt_conflict = self.monitor.analyze_gram_matrix(G)
        
        self.assertEqual(C.shape, (3, 3))
        self.assertAlmostEqual(C[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(C[1, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(C[2, 2].item(), 1.0, places=5)
        self.assertAlmostEqual(C[0, 1].item(), -1.0, places=5) # Complete clash
        self.assertAlmostEqual(C[0, 2].item(), 0.0, places=5)  # Orthogonal
        
        # Off diagonals: (-1.0, 0.0, -1.0, 0.0, 0.0, 0.0) -> mean = -2/6 = -0.3333
        self.assertAlmostEqual(mean_align, -1.0 / 3.0, places=4)
        # 2 out of 6 off diagonals are negative -> clash ratio = 2/6 = 0.3333
        self.assertAlmostEqual(clash_ratio, 1.0 / 3.0, places=4)
        print("[Test] Gram matrix & cosine similarity analytics validated (PASSED)")

    def test_speedup_benchmark(self):
        """Benchmark speedup of vmap vs iterative autograd on larger batch."""
        batch_size = 128
        x_batch = torch.randn(batch_size, self.in_dim, device=self.device)
        
        def dummy_loss(u, x):
            return torch.sum((u - x.sum()) ** 2)
            
        # Warmup
        _ = self.monitor.compute_per_point_gradients_vmap(self.mlp, x_batch, dummy_loss)
        _ = self.monitor.compute_per_point_gradients_autograd_fallback(self.mlp, x_batch[:8], dummy_loss)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.time()
        for _ in range(5):
            _ = self.monitor.compute_per_point_gradients_vmap(self.mlp, x_batch, dummy_loss)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_vmap = (time.time() - t0) / 5
        
        t0 = time.time()
        for _ in range(5):
            _ = self.monitor.compute_per_point_gradients_autograd_fallback(self.mlp, x_batch, dummy_loss)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_autograd = (time.time() - t0) / 5
        
        speedup = t_autograd / max(t_vmap, 1e-6)
        print(f"[Test] Batch size {batch_size} Profiler Speedup: vmap {t_vmap*1000:.2f} ms vs autograd {t_autograd*1000:.2f} ms -> {speedup:.1f}x speedup")
        self.assertGreater(speedup, 1.5)

if __name__ == "__main__":
    unittest.main()
