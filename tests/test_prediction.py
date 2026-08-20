import unittest

import torch

from genprior.prediction import fuse_distances


class FuseDistancesTest(unittest.TestCase):
    def test_normalizes_and_adds_distances(self):
        latent = torch.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 6.0]])
        visual = torch.tensor([[3.0, 1.0, 2.0], [2.0, 5.0, 1.0]])

        expected = (
            latent / (latent.mean(dim=1, keepdim=True) + 1e-5)
            + visual / (visual.mean(dim=1, keepdim=True) + 1e-5)
        )

        torch.testing.assert_close(fuse_distances(latent, visual), expected)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            fuse_distances(torch.ones(2, 3), torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
