from abc import ABC, abstractmethod

from torch import no_grad, eye, ones_like
from torch.nn import Module, ModuleList

from funcshape.layers.layerbase import DeepShapeLayer


class ShapeReparamBase(Module, ABC):
    def __init__(self, layerlist):
        super().__init__()
        self.layerlist = ModuleList(layerlist)
        for layer in layerlist:
            assert isinstance(
                layer, DeepShapeLayer
            ), "Layers must inherit DeepShapeLayer"

        self.project()

    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x

    @abstractmethod
    def derivative(self, x, h):
        pass

    def project(self, **kwargs):
        with no_grad():
            for module in self.modules():
                if isinstance(module, DeepShapeLayer):
                    module.project(**kwargs)

    def to(self, device):
        super().to(device)
        for module in self.modules():
            if isinstance(module, DeepShapeLayer):
                module.to(device)

        return self

    def attach(self):
        self._require_grad(True)

    def detach(self):
        self._require_grad(False)

    def _require_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad


class CurveReparametrizer(ShapeReparamBase):
    def derivative(self, x, h):
        dc = ones_like(x)
        for layer in self.layerlist:
            dc = dc * layer.derivative(x, h)
            x = layer(x)
        return dc


class SurfaceReparametrizer(ShapeReparamBase):
    def derivative(self, x, h):
        Df = eye(2, 2, device=x.device)
        for layer in self.layerlist:
            Df = layer.derivative(x, h) @ Df
            x = layer(x)
        return Df
