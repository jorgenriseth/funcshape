import re
from torch.nn import Module
from abc import ABC, abstractmethod

from funcshape.derivatives import central_differences, jacobian


class DeepShapeLayer(Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivative(self, x, h=1e-4):
        pass

    @abstractmethod
    def project(self):
        pass

    def to(self, device):
        super().to(device)

    def set_require_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad


class SurfaceLayer(DeepShapeLayer):
    def derivative(self, x, h=1e-4):
        return jacobian(self, x, h)


class CurveLayer(DeepShapeLayer):
    def derivative(self, x, h=1e-4):
        return central_differences(self, x, h)
