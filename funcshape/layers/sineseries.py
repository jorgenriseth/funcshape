import torch
from numpy import pi

from funcshape.layers.layerbase import CurveLayer

class SineSeries(CurveLayer):
    def __init__(self, N, init_scale=0., p=1):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1, dtype=torch.float)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.p = p
        self.project()

    def forward(self, x):
        return x + ((torch.sin(pi * self.nvec * x) / (pi * self.nvec)) @ self.weights)

    def derivative(self, x, h=None):
        return 1. + torch.cos(pi * self.nvec * x) @ self.weights

    def project(self):
        with torch.no_grad():
            norm = self.weights.norm(p=1)
            if norm > 1.0 - 1e-6:
                self.weights *= (1 - 1e-6) / norm

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self
