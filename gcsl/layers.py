"""Grouped SVD residuals and prefix-dependent basis orthogonality regularization."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualGroup(nn.Module):
    def __init__(self, u, s, v):
        super().__init__()
        self.u = nn.Parameter(u.clone())
        self.s = nn.Parameter(s.clone())
        self.v = nn.Parameter(v.clone())

    def forward(self, x):
        return F.linear(F.linear(x, self.v.T) * self.s, self.u)


class GCRLinear(nn.Module):
    """Freeze principal weights and train trailing spectral components grouped in singular-value order."""

    def __init__(self, linear, ranks=(4, 2, 1)):
        super().__init__()
        ranks = tuple(ranks)
        if not ranks or any(type(r) is not int or r <= 0 for r in ranks):
            raise ValueError("Residual ranks must be a nonempty sequence of positive integers")
        n = min(linear.weight.shape)
        if sum(ranks) >= n:
            raise ValueError("Total residual rank must be smaller than the minimum projection dimension")
        self.ranks = ranks
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # Compute SVD in float32 because half-precision decomposition is not supported.
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(linear.weight.detach().float(), full_matrices=False)
            offset = n - sum(ranks)
            self.register_buffer("u_main", u[:, :offset].clone())
            self.register_buffer("v_main", vh[:offset].T.clone())
            self.register_buffer("weight_main", (u[:, :offset] * s[:offset]) @ vh[:offset])
            self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().float().clone())
            eye = torch.eye(offset, device=u.device)
            self.register_buffer("main_gram_error", (
                (self.u_main.T @ self.u_main - eye).square().sum()
                + (self.v_main.T @ self.v_main - eye).square().sum()))
            self.groups = nn.ModuleList()
            for rank in ranks:
                end = offset + rank
                self.groups.append(ResidualGroup(u[:, offset:end], s[offset:end], vh[offset:end].T))
                offset = end
        self.prefix = len(ranks)

    def set_prefix(self, prefix):
        if not 1 <= prefix <= len(self.groups):
            raise ValueError("Prefix depth is outside the residual-group range")
        self.prefix = prefix

    def forward(self, x):
        # Use out-of-place additions so forward passes never modify frozen weights.
        result = F.linear(x, self.weight_main, self.bias)
        for group in self.groups[:self.prefix]:
            result = result + group(x)
        return result

    def orthogonality_loss(self):
        """Normalize the two squared Gram errors by 2q squared, following the paper."""
        active = self.groups[:self.prefix]
        u = torch.cat([g.u.float() for g in active], dim=1)
        v = torch.cat([g.v.float() for g in active], dim=1)
        q = self.u_main.shape[1] + u.shape[1]
        # Evaluate the full Gram loss in blocks, including numerical error in the frozen principal block.
        def penalty(main, residual):
            main = main.float()
            eye_res = torch.eye(residual.shape[1], device=main.device)
            return (2 * (main.T @ residual).square().sum()
                    + (residual.T @ residual - eye_res).square().sum())
        return (self.main_gram_error + penalty(self.u_main, u) + penalty(self.v_main, v)) / (2 * q * q)
