import torch
import matplotlib.pyplot as plt
from numpy import pi
from torch import sin, cos, zeros_like

from funcshape.surface import Surface
from funcshape.visual import get_plot_data
from savefig import savefig

FIELD_INFO = [
    dict(type=1, k=1, l=0, direction="x"),
    dict(type=3, k=1, l=1, direction="x"),
    dict(type=2, k=1, l=1, direction="y"),
    dict(type=2, k=3, l=1, direction="y"),

]

def example_vector_fields(savename, field_info, figblock=True):
    npoints = 24
    N = len(field_info)
    fig, axes = plt.subplots(1, N, figsize=(12, 12 / N))
    for idx, field in enumerate(field_info):
        v = create_basisfield(**field)
        plot_quiver(v, npoints, axes[idx])
    plt.tight_layout()
    for i, caption in enumerate([r"$\xi_1$", r"$\varphi_{1,1}$", r"$\tilde\eta_{1, 1}$", r"$\tilde\eta_{1, 3}$"]):
        axes[i].text(0.4, -0.3, caption, fontsize=16)
    savefig(savename, fig)
    plt.show(block=figblock)  

def plot_quiver(v, npoints, ax=None,  **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    X, Y = torch.meshgrid((torch.linspace(0, 1, npoints), torch.linspace(0, 1, npoints)))
    Z = get_plot_data(v, k=npoints)
    ax.quiver(X, Y, Z[0], Z[1], **kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def create_basisfield(type, k, l, direction):
    if type == 1:
        return BasisVectorFieldType1(k, direction)
    elif type == 2:
        return BasisVectorFieldType2(k, l, direction)
    elif type == 3:
        return BasisVectorFieldType3(k, l, direction)
    else:
        raise ValueError("type should be 1, 2 or 3, got: ", type)


class BasisVectorFieldType1(Surface):
    def __init__(self, k: int, direction: str):
        if direction == "x":
            components = (
                lambda x: sin(pi * k * x[..., 0]) / (pi * k),
                lambda x: zeros_like(x[..., 0]),
            )
        elif direction == "y":
            components = (
                lambda x: zeros_like(x[..., 0]),
                lambda x: sin(pi * k * x[..., 1]) / (pi * k),
            )
        else:
            raise ValueError("Direction must be 'x' or 'y', got:", direction)
        super().__init__(components)


class BasisVectorFieldType2(Surface):
    def __init__(self, k: int, l: int, direction: str):
        if direction == "x":
            components = (
                lambda x: sin(pi * k * x[..., 0]) * cos(2 * pi * l * x[..., 1]) / (pi * k * l),
                lambda x: zeros_like(x[..., 0]),
            )
        elif direction == "y":
            components = (
                lambda x: zeros_like(x[..., 0]),
                lambda x: sin(pi*k*x[..., 1]) * cos(2*pi*l*x[..., 0]) / (pi * k * l),
            )
        else:
            raise ValueError("Direction must be 'x' or 'y', got:", direction)
        super().__init__(components)


class BasisVectorFieldType3(Surface):
    def __init__(self, k: int, l: int, direction: str):
        if direction == "x":
            components = (
                lambda x: sin(pi*k*x[..., 0]) * sin(2*pi*l*x[..., 1]) / (pi*k*l),
                lambda x: zeros_like(x[..., 0]),
            )
        elif direction == "y":
            components = (
                lambda x: zeros_like(x[..., 0]),
                lambda x: sin(pi*k*x[..., 1]) * sin(2*pi*l*x[..., 0]) / (pi*k*l),
            )
        else:
            raise ValueError("Direction must be 'x' or 'y', got:", direction)
        super().__init__(components)


if __name__ =="__main__":
    example_vector_fields("Fig3.eps", FIELD_INFO, False)