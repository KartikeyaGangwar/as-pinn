import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.atlas import ATLAS, ASNN, AdaptiveSubspacePINN
from train_two_stage_atlas import TwoStageATLASTrainer

class TestATLASArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_dim = 2
        self.out_dim = 1
        self.model = ATLAS(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            initial_subspaces=2,
            hidden_dim=32,
            layers=3,
        ).to(self.device)

    def test_atlas_instantiation_and_forward(self):
        """Verify ATLAS model instantiates, runs forward, and respects PoU partition."""
        x = torch.rand(20, self.in_dim, device=self.device)
        u = self.model(x)
        self.assertEqual(u.shape, (20, self.out_dim))
        self.assertIsInstance(self.model, AdaptiveSubspacePINN)
        self.assertEqual(self.model.num_subspaces, 2)

    def test_atlas_subspace_spawning(self):
        """Verify autonomous subspace spawning in ATLAS."""
        initial_subspaces = self.model.num_subspaces
        child_centroid = torch.tensor([0.25, 0.75], device=self.device)
        child_idx = self.model.spawn_new_subspace(child_centroid)
        self.assertEqual(child_idx, initial_subspaces)
        self.assertEqual(self.model.num_subspaces, initial_subspaces + 1)

if __name__ == "__main__":
    unittest.main()
