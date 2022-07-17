import torch
from funcshape.curve import Curve
from funcshape.surface import Surface

class Qmap1D:
    """ Q-map transformation of curves """
    def __init__(self, curve):
        self.c = curve

    def __call__(self, X, h=1e-4):
        return torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)) * self.c(X)


class SRVT:
    """ SRVT of curves """
    def __init__(self, curve: Curve):
        super().__init__()
        self.c = curve

    def __call__(self, X, h=1e-4):
        u = self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)
        return torch.where(
            torch.abs(u) < 1e-7,
            torch.zeros((u.shape[0], self.c.dim)),
            self.c.derivative(
                X, h=h) / torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True))
        )

    def inverse(self, X):
        Q = self(X)
        h = 1. / (X.shape[0] - 1)
        points = Q * Q.norm(dim=-1, keepdim=True)
        return h * points.cumsum(dim=0)

    def compose(self, f):
        return SRVT(self.c.compose(f))


class Qmap2D:
    def __init__(self, surface):
        self.s = surface

    def __call__(self, X, h=1e-4):
        return torch.sqrt(self.s.volume_factor(X, h)) * self.s(X)


class SRNF:
    def __init__(self, surface):
        self.s = surface

    def __call__(self, X, h=1e-4):
        n = self.s.normal_vector(X)
        u = torch.norm(n, dim=-1, keepdim=True)
        return torch.where(
            torch.abs(u) < 1e-7,
            torch.zeros((u.shape[0], self.s.dim), device=X.device),
            torch.sqrt(self.s.volume_factor(X, h)) * n / torch.norm(n, dim=-1, keepdim=True)
        )
