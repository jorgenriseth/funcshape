import numpy as np
import torch

from funcshape.diffeomorphism import Diffeomorphism2D
from funcshape.surface import Surface

class CylinderWrap(Surface):
    def __init__(self):
        super().__init__((
            lambda x: torch.sin(2*np.pi*x[..., 0]),
            lambda x: torch.sin(4*np.pi*x[..., 0]),
            lambda x: x[..., 1]
        ))


class HyperbolicParaboloid(Surface):
    def __init__(self):
        super().__init__((
            lambda x: x[..., 0],
            lambda x: x[..., 1],
            lambda x: (x[..., 0] - 0.5)**2 - (x[..., 1] - 0.5)**2
        ))


class LogStepQuadratic(Diffeomorphism2D):
    def __init__(self, a=20., b=0.1):
        assert 0. < b < 1., "b must be between 0 and 1."
        c = (1 - b)
        super().__init__((
            lambda x: c * x[..., 0]**2 + b * x[..., 0],
            lambda x: (0.5 * torch.log(a*x[..., 1]+1) / torch.log(21*torch.ones(1))
                       + 0.25 * (1 + torch.tanh(a*(x[..., 1]-0.5)) / torch.tanh(21*torch.ones(1))))
        ))


# Helper function to create Rotation-Diffeomorphism
def angle(x):
    return 0.5 * np.pi * torch.sin(np.pi * x[..., 0]) * torch.sin(np.pi * x[..., 1])


class RotationDiffeomorphism(Diffeomorphism2D):
    def __init__(self):
        super().__init__((
            lambda x: (x[..., 0] - 0.5) * torch.cos(angle(x)) -
            (x[..., 1] - 0.5) * torch.sin(angle(x)) + 0.5,
            lambda x: (x[..., 0] - 0.5) * torch.sin(angle(x)) +
            (x[..., 1] - 0.5) * torch.cos(angle(x)) + 0.5
        ))
