from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize

from funcshape.networks import CurveReparametrizer
from funcshape.layers.sineseries import SineSeries
from funcshape.loss import CurveDistance
from funcshape.logging import Logger
from funcshape.reparametrize import reparametrize
from funcshape.utils import col_linspace
from funcshape.visual import plot_curve
from funcshape.interpolation import linear_interpolate, geodesic
from funcshape.testlib.curves import HalfCircle, Circle
from funcshape.visual import create_figsaver
from funcshape.testlib.curves import Circle
from funcshape.transforms import SRVT


def plot_curve_interpolation(savename, figblock=True, colormap='jet'):
    savefig = create_figsaver(Path(__file__).parent / "../figures/")

    c1 = Circle()
    c2 = HalfCircle()
    q = SRVT(c1)
    r = SRVT(c2)

    RN = CurveReparametrizer([
        SineSeries(10) for i in range(4)
    ])

    # Define loss function
    loss_func = CurveDistance(q, r, k=2048)
    optimizer = optim.LBFGS(RN.parameters(), lr=1, max_iter=100, line_search_fn='strong_wolfe')
    error = reparametrize(RN, loss_func, optimizer, 300, Logger(1))

    # Get plot data to visualize diffeomorphism
    RN.detach()
    x = col_linspace(0, 1, 1024)
    y = RN(x)
    rafter = r.compose(RN)

    h = 1e-3
    # Interpolate
    numsteps=10
    curves = linear_interpolate(c1, c2, steps=numsteps)
    curves_after = linear_interpolate(c1, c2.compose(RN), steps=numsteps)
    srvts = geodesic(lambda x: q(x, h=h), lambda x: rafter(x, h=h), steps=numsteps)
    colors = matplotlib.colormaps[colormap](np.linspace(0, 1, numsteps))  # Change colormap here.

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    [plot_curve(ci, ax=ax1, color=colori, npoints=601) for ci, colori in zip(curves, colors)]
    ax1.set_aspect("equal")
    ax1.set_xticks([-1, 1])
    ax1.set_yticks([-1, 1])


    [plot_curve(ci, ax=ax2, color=colori, npoints=601) for ci, colori in zip(curves_after, colors)]
    ax2.set_aspect("equal")
    ax2.set_xticks([-1, 1])
    ax2.set_yticks([-1, 1])


    start = torch.tensor([1., 0.])
    [plot_curve(lambda x: ci(x) + start, ax=ax3, color=colori, npoints=601) for ci, colori in zip(srvts, colors)]
    ax3.set_aspect("equal")
    ax3.set_xticks([-1, 1])
    ax3.set_yticks([-1, 1])

    norm = Normalize(0, 1)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax)
    cbar_ax.text(-1.8, 0.45, r"$\tau$", fontsize=16)
    savefig(savename)
    if figblock:
        plt.show(block=figblock)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    print()

    plot_curve_interpolation("Fig5.eps", args.show, "inferno")
