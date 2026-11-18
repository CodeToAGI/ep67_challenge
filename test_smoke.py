import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ep67_mamba", ROOT / "ep67_mamba.py")
episode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(episode)


class EpisodeSmokeTests(unittest.TestCase):
    def test_recurrence_matches_convolution(self):
        result = episode.equivalence(L=64, N=4, seed=0)
        self.assertLess(result["max_diff"], 1e-5)

    def test_all_models_return_expected_shape(self):
        generator = torch.Generator().manual_seed(123)
        x, targets = episode.batch(B=2, L=32, g=generator)
        self.assertEqual(tuple(targets.shape), (2, episode.M))

        for kind in episode.KINDS:
            with self.subTest(kind=kind):
                model = episode.Net(kind)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, episode.M, episode.V + 1))
                self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
