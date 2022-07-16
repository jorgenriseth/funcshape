import torch

from funcshape.derivatives import central_differences

class Curve:
    """ Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator. """
    def __init__(self, component_function_tuple):
        self.C = tuple(component_function_tuple)
        self.dim = len(self.C)

    def __call__(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)

    def derivative(self, X, h=1e-4):
        return torch.cat([central_differences(ci, X, h) for ci in self.C], dim=-1)

    def compose_component(self, i, f):
        return lambda x: self.C[i](f(x))

    def compose(self, f):
        return Curve((self.compose_component(i, f) for i in range(self.dim)))
