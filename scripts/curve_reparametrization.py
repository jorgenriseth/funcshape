from pathlib import Path

import torch.optim as optim
import matplotlib.pyplot as plt

from funcshape.networks import CurveReparametrizer
from funcshape.layers.sineseries import SineSeries
from funcshape.loss import CurveDistance
from funcshape.logging import Logger
from funcshape.testlib.curves import Infinity, LogStepDiff, TorchCircle
from funcshape.transforms import Qmap1D
from funcshape.reparametrize import reparametrize
from funcshape.utils import col_linspace
from funcshape.visual import plot_curve


from savefig import savefig
import torch

torch.set_default_dtype(torch.float64)


from funcshape.curve import ComposedCurve


def plot_curve_reparametrization(savename, figblock, h):
    # Analytic diffeomorphism
    g = LogStepDiff()

    # Define Curves
    c1 = Infinity()
    # c1 = TorchCircle()
    # c2 = c1.compose(g)
    c0 = ComposedCurve(c1, g)

    # Get Qmaps (reparametrize c1 into c0(x) = c1(g(x)))
    q, r = Qmap1D(c0), Qmap1D(c1)

    # Create reparametrization network
    RN = CurveReparametrizer([SineSeries(20) for i in range(20)])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    RN.to(device)
    loss_func = CurveDistance(q, r, k=1024, h=None).to(device)

    # Define loss, optimizer and run reparametrization.
    # loss_func = CurveDistance(q, r, k=1024, h=None)
    optimizer = optim.LBFGS(
        RN.parameters(),
        max_iter=500,
        max_eval=3 * 500,
        history_size=500,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
    )
    error = reparametrize(RN, loss_func, optimizer, None, Logger(0))
    print(error[-1])

    # Get plot data to visualize diffeomorphism
    RN.to("cpu")
    RN.detach()
    x = col_linspace(0, 1, 2048)
    y = RN(x)

    # Get curve-coordinates before and after reparametrization
    C1, C2, C3 = c1(x), c0(x), c1(y)

    # fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = fig.add_gridspec(4, 3)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2:, 0])
    plot_curve(c1, dotpoints=41, ax=ax1)
    plot_curve(c0, dotpoints=41, ax=ax2)

    # Plot coordinates before reparametrization
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[3, 1])

    ax3.plot(x, C2[:, 0], ls="dashed")
    ax3.plot(x, C1[:, 0])
    ax4.plot(x, C2[:, 1], ls="dashed")
    ax4.plot(x, C1[:, 1])

    # Plot coordinates after reparametrization
    ax5.plot(x, C2[:, 0], ls="dashed")
    ax5.plot(x, C3[:, 0])
    ax6.plot(x, C2[:, 1], ls="dashed")
    ax6.plot(x, C3[:, 1])

    ax7 = fig.add_subplot(gs[:2, 2])
    ax8 = fig.add_subplot(gs[2:, 2])
    ax7.plot(x, y, lw=2)
    ax7.plot(x, g(x), ls="--", c="black", dashes=(5, 5))
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax8.semilogy(error / error[0])

    plt.tight_layout()
    savefig(savename)
    plt.show(block=figblock)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    print()
    plot_curve_reparametrization("Fig6.eps", args.show, 1e-3)
    plt.show()
