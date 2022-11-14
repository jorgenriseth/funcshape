from math import gamma
import time
import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from funcshape.testlib.surfaces import HyperbolicParaboloid, RotationDiffeomorphism, CylinderWrap, LogStepQuadratic

from funcshape.transforms import Qmap2D, SRNF
from funcshape.visual import (
    get_common_colornorm,
    plot_surface,
    plot_diffeomorphism_2d
)
from funcshape.networks import SurfaceReparametrizer
from funcshape.layers.sinefourier import SineFourierLayer
from funcshape.loss import SurfaceDistance
from funcshape.reparametrize import reparametrize
from funcshape.logging import Logger

from savefig import savefig

def setup_case1():
    f = HyperbolicParaboloid()
    γ = RotationDiffeomorphism()
    g = f.compose(γ)
    q = SRNF(g)
    r = SRNF(f)

    RN = SurfaceReparametrizer(
        [SineFourierLayer(10) for _ in range(10)]
    )
    return f, γ, g, q, r, RN

def setup_case2():
    f = CylinderWrap()
    γ = RotationDiffeomorphism().compose(LogStepQuadratic())
    g = f.compose(γ)
    q = Qmap2D(g)
    r = Qmap2D(f)

    # Define reparametrization-network
    RN = SurfaceReparametrizer(
        [SineFourierLayer(20) for _ in range(20)]
    )

    return f, γ, g, q, r, RN



def reparametrize_and_plot(savename, setup_case, figblock=False):
    f, γ, g, q, r, RN = setup_case()

    # Use GPU capabilities? Makes small differences for small networks,
    # but is significantly more scalable.
    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    RN.to(device)
    loss_func = SurfaceDistance(q, r, k=32, h=None).to(device)

    optimizer = optim.LBFGS(RN.parameters(), max_iter=30, line_search_fn="strong_wolfe")
    errors = reparametrize(RN, loss_func, optimizer, 1, Logger(1))
    RN.to("cpu"), loss_func.to("cpu"); # Need on CPU for plotting.
    RN.detach()

    # Reparametrize f
    fafter = f.compose(RN)

    # Create Coloring Functions for plotting
    k = 256 # Points per dimension (k^2 points total)
    camera = (35, 225)
    norm = get_common_colornorm((f, g, fafter), k=k)

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 6)

    ax1 = fig.add_subplot(gs[0, :2], projection="3d")
    ax2 = fig.add_subplot(gs[0, 2:4], projection="3d")
    ax3 = fig.add_subplot(gs[0, 4:], projection="3d")

    plot_surface(g, ax=ax1, k=k, colornorm=norm, camera=camera)
    # ax1.set_title("Target Surface")
    ax1.set_title(r"(a) Target surface $f \circ \varphi$", fontsize=16)
    plot_surface(f, ax=ax2, k=k, colornorm=norm, camera=camera)
    # ax2.set_title("Subject Surface")
    ax2.set_title(r"(b) Subject surface  $f$", fontsize=16)
    plot_surface(fafter, ax=ax3, k=k, colornorm=norm, camera=camera)
    ax3.set_title(r"(c) Reparametrized surface $f \circ \psi$", fontsize=16)

    # Plot Diffeomorphisms. 
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])

    plot_diffeomorphism_2d(γ, k=20, color="k", ax=ax4)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title(r"(d) True reparametrization $\varphi$", fontsize=16)

    plot_diffeomorphism_2d(RN, k=20, color="k", ax=ax5)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title(r"(e) Found reparametrization $\psi$", fontsize=16)
    plt.tight_layout()
    savefig(savename, fig)

    plt.figure()
    plt.semilogy(errors)
    plt.axhline(0., ls="--", c="black")
    plt.ylabel("Error", fontsize=16)
    plt.xlabel("Iteration", fontsize=16)
    plt.title("Convergence Plot")
    plt.show(block=figblock)  


if __name__ == "__main__":
    reparametrize_and_plot("Fig7.png", setup_case1, False)
    reparametrize_and_plot("Fig8.png", setup_case2, False)