from abc import ABC, abstractmethod

import torch

from funcshape.derivatives import batch_determinant
from funcshape.utils import torch_square_grid, col_linspace#, component_mse


class ShapeDistanceBase(ABC):
    @abstractmethod
    def create_point_collection(self, k):
        pass

    @abstractmethod
    def get_determinant(self, network):
        pass

    def __init__(self, q, r, k, h=1e-4):
        self.h = h
        self.X = self.create_point_collection(k)
        self.k = k**(self.X.shape[-1])
        self.r = r
        self.Q = q(self.X)

    def loss_func(self, U, Y):
        return ((self.Q - torch.sqrt(U+1e-7) * self.r(Y))**2).sum() / self.k

    def get_last(self):
        return self.loss

    def to(self, device):
        self.X = self.X.to(device)
        self.Q = self.Q.to(device)
        return self

    def __call__(self, network):
        Y = network(self.X)
        U = self.get_determinant(network)

        # Check for invalid derivatives. Retry projection, or raise error.
        if U.min() < 0. or torch.isnan(U.min()):
            network.project()
            Y = network(self.X)
            U = self.get_determinant(network)
            if U.min() < 0. or torch.isnan(U.min()):
                raise ValueError(
                    f"ProjectionError: derivative minimum is {float(U.min())}")

        loss = self.loss_func(U, Y)
        self.loss = float(loss)
        return loss


class CurveDistance(ShapeDistanceBase):
    def create_point_collection(self, k):
        return col_linspace(0, 1, k)

    def get_determinant(self, network):
        return network.derivative(self.X, self.h)


class SurfaceDistance(ShapeDistanceBase):
    def create_point_collection(self, k):
        return torch_square_grid(k).reshape(-1, 2)

    def get_determinant(self, network):
        return batch_determinant(network.derivative(self.X, self.h))


class ComponentDistance(ShapeDistanceBase):
    def __init__(self, q, r, k, h=1e-4, component=-1):
        super().__init__(q, r, k, h=h)
        self.component = component

    def loss_func(self, U, Y):
        return component_mse(self.Q, torch.sqrt(U+1e-8) * self.r(Y), self.component)



def component_mse(inputs, targets, component: int):
    """ Stored here for now, will probably be moved elsewhere in the future"""
    return torch.sum((inputs[..., component] - targets[..., component])**2) / inputs[..., component].nelement()


class ImageComponentDistance(SurfaceDistance, ComponentDistance):
    pass
