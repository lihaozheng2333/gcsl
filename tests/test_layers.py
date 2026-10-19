"""Verify weight preservation, gradient selection, and the regularization formula."""

import unittest

import torch
from torch import nn

from gcsl.layers import GCRLinear


class SubspaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.original = nn.Linear(12, 12)
        self.layer = GCRLinear(self.original)

    def test_full_prefix_preserves_pretrained_projection(self):
        x = torch.randn(5, 12)
        main = self.layer.weight_main.clone()
        for _ in range(3):
            torch.testing.assert_close(self.layer(x), self.original(x), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(main, self.layer.weight_main, atol=0, rtol=0)

    def test_inactive_groups_do_not_receive_gradients(self):
        self.layer.set_prefix(1)
        (self.layer(torch.randn(5, 12)).square().mean()
         + self.layer.orthogonality_loss()).backward()
        self.assertIsNotNone(self.layer.groups[0].u.grad)
        for group in self.layer.groups[1:]:
            self.assertTrue(all(p.grad is None for p in group.parameters()))

    def test_regularizer_matches_full_gram_formula(self):
        with torch.no_grad():
            self.layer.groups[0].u.add_(0.1)
        for prefix in (1, 2, 3):
            self.layer.set_prefix(prefix)
            u = torch.cat([self.layer.u_main] + [g.u for g in self.layer.groups[:prefix]], 1)
            v = torch.cat([self.layer.v_main] + [g.v for g in self.layer.groups[:prefix]], 1)
            q = u.shape[1]
            eye = torch.eye(q)
            direct = ((u.T @ u - eye).square().sum() + (v.T @ v - eye).square().sum()) / (2 * q*q)
            torch.testing.assert_close(self.layer.orthogonality_loss(), direct)

    def test_invalid_rank_is_rejected(self):
        for ranks in ([], [0], [-1], [12]):
            with self.assertRaises(ValueError):
                GCRLinear(self.original, ranks)


if __name__ == "__main__":
    unittest.main()
