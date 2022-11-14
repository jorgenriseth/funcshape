from pathlib import Path

import torch.optim as optim
import matplotlib.pyplot as plt

from funcshape.networks import CurveReparametrizer
from funcshape.layers.sineseries import SineSeries
from funcshape.loss import CurveDistance
from funcshape.logging import Logger
from funcshape.testlib.curves import Infinity, LogStepDiff
from funcshape.transforms import Qmap1D
from funcshape.reparametrize import reparametrize
from funcshape.utils import col_linspace
from funcshape.visual import plot_curve


from savefig import savefig

def plot_curve_reparametrization(savename, figblock):
    # Analytic diffeomorphism
    g = LogStepDiff()

    # Define Curves 
    c1 = Infinity()
    c2 = c1.compose(g)

    # Get Qmaps (reparametrize c1 into c2(x) = c1(g(x)))
    q, r = Qmap1D(c2), Qmap1D(c1)


    # Create reparametrization network
    RN = CurveReparametrizer([
        SineSeries(10) for i in range(10)
    ])

    # Define loss, optimizer and run reparametrization.
    loss_func = CurveDistance(q, r, k=1024)
    optimizer = optim.LBFGS(RN.parameters(), lr=1., max_iter=200, line_search_fn='strong_wolfe')
    error = reparametrize(RN, loss_func, optimizer, 200, Logger(0))

    # Get plot data to visualize diffeomorphism
    RN.detach()
    x = col_linspace(0, 1, 1024)
    y = RN(x)

    # Get curve-coordinates before and after reparametrization
    C1, C2, C3 = c1(x), c2(x), c1(y)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = fig.add_gridspec(4, 3)

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2:, 0])
    plot_curve(c1, dotpoints=41, ax=ax1)
    plot_curve(c2, dotpoints=41, ax=ax2)

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
    ax8.semilogy(error / error[0] )

    plt.tight_layout()
    savefig(savename)
    plt.show(block=figblock)

if __name__ == "__main__":
    plot_curve_reparametrization("Fig6.eps", False)
    
