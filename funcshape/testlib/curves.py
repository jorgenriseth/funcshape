import torch
import numpy as np
from numpy import pi

from funcshape.curve import Curve
from funcshape.diffeomorphism import Diffeomorphism1D


class Circle(Curve):
    def __init__(self):
        super().__init__((
            lambda x: torch.cos(2*pi*x),
            lambda x: torch.sin(2*pi*x)
        ))


class Infinity(Curve):
    def __init__(self):
        super().__init__((
            lambda x: torch.cos(2*pi*x),
            lambda x: torch.sin(4*pi*x)
        ))


class HalfCircle(Curve):
    def __init__(self, transform="qmap"):
        if transform.lower() == "qmap":
            super().__init__((
                lambda x: torch.cos(pi * x),# / (pi), #/ np.cbrt(pi),
                lambda x: torch.sin(pi * x)# / (pi) #/ np.cbrt(pi)
            ))
        elif transform.lower() == "srvt":
            super().__init__((
                lambda x: torch.sin(pi * x) / pi,
                lambda x: -torch.cos(pi * x) / pi
            ))
        else:
            raise ValueError("invalid transform. Must be 'srvt' or 'qmap'")


class Line(Curve):
    def __init__(self, transform="qmap"):
        if transform.lower() == "qmap":
            super().__init__((
                lambda x: torch.zeros_like(x),
                lambda x: torch.pow(3*x + 1., (1./3.))
            ))
        elif transform.lower() == "srvt":
            super().__init__((
                lambda x: torch.ones_like(x),
                lambda x: x - 0.5
            ))
        else:
            raise ValueError("invalid transform. Must be 'srvt' or 'qmap'")


class Id(Diffeomorphism1D):
    def __init__(self):
        super().__init__(lambda x: x)


class QuadDiff(Diffeomorphism1D):
    def __init__(self, b=0.1):
        assert 0. < b < 2., "b should be between 0 and 2"
        c = 1. - b
        super().__init__(lambda x: c * x**2 + b * x)


class LogStepDiff(Diffeomorphism1D):
    def __init__(self, a=20):
        super().__init__(
            lambda x: (0.5 * torch.log(a*x+1) / np.log(a+1.)
                       + 0.25 * (1 + torch.tanh(a*(x-0.5)) / np.tanh(a+1.)))
        )


class OptimalCircleLine(Diffeomorphism1D):
    def __init__(self):
        super().__init__(lambda x: x - 0.5 * torch.sin(2*pi*x) / pi)


class EdgeCase1(Diffeomorphism1D):
    def __init__(self):
        super().__init__(lambda x: 2. * x * (x < 0.5) + 1. * (x >= 0.5))

 
class EdgeCase2(Diffeomorphism1D):
    def __init__(self):
        super().__init__(lambda x: 0.5 * x * (x < 1.) + 1. * (x >= 1.))
