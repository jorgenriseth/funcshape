from pathlib import Path

import numpy as np
import torch
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

    
def gradient_descent_comparison(savename, block=True):
    from funcshape.testlib.curves import Circle, LogStepDiff
    from funcshape.transforms import SRVT
    from funcshape.gradient_descent import gradient_descent, backtracking_linesearch

    # Analytic diffeomorphism
    c2 = Circle()
    g = LogStepDiff()
    c1 = c2.compose(g)
    q = SRVT(c1)
    r = SRVT(c2)

    n = 6  # Number of basis functions in layer
    k = 1024  # Number of points used for evaluation.


    # Deep reparametrization
    loss = CurveDistance(q, r, k=k)
    RN = CurveReparametrizer([
        SineSeries(n) for _ in range(6)
    ])
    opt = torch.optim.LBFGS(RN.parameters(), line_search_fn='strong_wolfe', max_iter=300, max_eval=300)
    error_deep = reparametrize(RN, loss, opt, 1, Logger(1))
    RN.detach()
    x = col_linspace(0, 1, k)
    y_deep = RN(x)

    # Gradient Descent
    y_gd, error_gd = gradient_descent(q, r, k, n, verbose=True)

    # Plot curves
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 0.75]})
    plot_curve(c1, npoints=k, dotpoints=41, ax=axes[0][0])
    plot_curve(c2, npoints=k, dotpoints=41, ax=axes[0][1])
    for ax in axes[0]:
        ax.set_aspect("equal")
    # plt.show()

    # Plot diffeomorphism and subplot
    # fig, (ax[1][1], ax[1][1]) = plt.subplots(1, 2, figsize=(10, 4))
    axes[1][0].plot(x, y_deep, label="Deep")
    axes[1][0].plot(x, y_gd, label="GD")
    axes[1][0].plot(x, g(x), 'k--', label=r'$ \varphi $')
    axes[1][0].set_xlim(0, 1)
    axes[1][0].set_ylim(0, 1)
    axes[1][0].legend()
    axes[1][1].semilogy(error_deep / error_deep[0])
    axes[1][1].semilogy(error_gd / error_gd[0])
    axes[1][1].set_xlim(0, None)
    plt.tight_layout()
    plt.tight_layout()
    savefig(savename, fig)

if __name__ == "__main__":
    gradient_descent_comparison("Fig1.eps", block=False)
