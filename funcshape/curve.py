import torch

from typing import Callable

from funcshape.derivatives import central_differences
from funcshape.diffeomorphism import Diffeomorphism1D

class Curve:
    """ Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator. """
    def __init__(self, component_function_tuple):
        self.C = tuple(component_function_tuple)
        self.dim = len(self.C)

    def __call__(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)

    def derivative(self, X, h):
        return torch.cat([central_differences(ci, X, h) for ci in self.C], dim=-1)

    def compose_component(self, i, f):
        return lambda x: self.C[i](f(x))

    def compose(self, f):
        return Curve((self.compose_component(i, f) for i in range(self.dim)))

class TorchCurve:
    def __init__(self, components: list[Callable]):
        self.components = components

    def __call__(self, x):
        return torch.stack([
            ci(x) for ci in self.components
        ], dim=-1)
    
    def derivative(self, x, h):
        return torch.stack([
            torch.autograd.grad(ci(x), x, torch.ones_like(x))[0] for ci in self.components
        ], dim=-1)       


class ComposedCurve(Curve):
    def __init__(self, curve: Curve, diffeomorphism: Diffeomorphism1D):
        self.c = curve
        self.diffeo = diffeomorphism

    def __call__(self, X):
        return self.c(self.diffeo(X))

    def derivative(self, X, h):
        return self.c.derivative(self.diffeo(X), h) * self.diffeo.derivative(X, h)

    @property
    def dim(self):
        return self.c.dim