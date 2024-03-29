import json

import torch
import matplotlib.pyplot as plt

from funcshape.curve import ComposedCurve
from funcshape.testlib.curves import Infinity, LogStepDiff, Circle
from funcshape.transforms import Qmap1D, SRVT, Qmap2D, SRNF
from funcshape.reparametrize import reparametrize
from funcshape.layers.sineseries import SineSeries
from funcshape.layers.sinefourier import SineFourierLayer
from funcshape.networks import CurveReparametrizer, SurfaceReparametrizer
from funcshape.loss import CurveDistance, SurfaceDistance
from funcshape.logging import Silent
from funcshape.visual import create_figsaver, MONOCHROME
from funcshape.testlib.surfaces import RotationDiffeomorphism, HyperbolicParaboloid

from savefig import savefig

import torch
torch.set_default_dtype(torch.float64)


def curve_reparametrization(
    c1,
    c2,
    num_layers,
    num_functions,
    k,
    max_iter,
    transform="qmap",
    device="cuda",
    **kwargs,
):
    print(f"Layers: {num_layers:4d}, functions: {num_functions:3d}\r", end="")
    if device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            raise RuntimeError("cuda not available")
    if transform.lower() == "qmap":
        q, r = Qmap1D(c1), Qmap1D(c2)
    elif transform.lower() == "srvt":
        q, r = SRVT(c1), SRVT(c2)
    else:
        raise ValueError("Transform should be 'qmap' or 'srvt'")

    RN = CurveReparametrizer([SineSeries(num_functions) for _ in range(num_layers)])
    RN.to(device)
    loss = CurveDistance(q, r, k=1024, h=None).to(device)
    optimizer = torch.optim.LBFGS(
        RN.parameters(),
        max_iter=max_iter,
        max_eval=3*max_iter,
        history_size=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )
    error = reparametrize(RN, loss, optimizer, None, **kwargs)
    RN.to("cpu"), loss.to("cpu")
    RN.detach()
    # Need on CPU for plotting.
    return error[-1]


def surface_reparametrization(
    c1,
    c2,
    num_layers,
    num_functions,
    transform="qmap",
    k=32,
    max_iter=200,
    device="cpu",
    **kwargs,
):
    print(f"Layers: {num_layers:4d}, functions: {num_functions:3d}\r", end="")
    if device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            raise RuntimeError("cuda not available")

    if transform.lower() == "qmap":
        q, r = Qmap2D(c1), Qmap2D(c2)
    elif transform.lower() == "SRNF":
        q, r = SRNF(c1), SRNF(c2)
    else:
        raise ValueError("Transform should be 'qmap' or 'srvt'")

    RN = SurfaceReparametrizer(
        [SineFourierLayer(num_functions) for _ in range(num_layers)]
    ).to(device)
    optimizer = torch.optim.LBFGS(
        RN.parameters(),
        max_iter=max_iter,
        max_eval=3*max_iter,
        history_size=max_iter,
        line_search_fn="strong_wolfe",
    )
    loss = SurfaceDistance(q, r, k=k).to(device)
    error = reparametrize(RN, loss, optimizer, 1, **kwargs)
    RN.to("cpu"), loss.to("cpu") # Need on CPU for plotting.
    return error[-1]


def create_convergence_dict(
    c0, c1, num_layers, num_functions, reparam_wrapper, **kwargs
):
    return {
        i: {j: reparam_wrapper(c0, c1, i, j, **kwargs) for j in num_functions}
        for i in num_layers
    }


def plot_depth_convergence(
    d, ax=None, subset=None, log=True, label_identifier="M", **kwargs
):
    E = depth_convergence(d)
    N = list(width_convergence(d))

    if ax is None:
        fig, ax = plt.subplots()

    for num_funcs, error in E.items():
        if num_funcs in subset or subset is None:
            if log:
                ax.semilogy(
                    N, error, label=f"${label_identifier}={num_funcs}$", **kwargs
                )
            else:
                ax.plot(N, error, label=f"${label_identifier}={num_funcs}$", **kwargs)
    ax.set_xticks(N)
    return ax


def plot_width_convergence(d, ax=None, subset=None, log=True, **kwargs):
    E = width_convergence(d)
    N = list(depth_convergence(d))

    if ax is None:
        fig, ax = plt.subplots()
    for num_layers, error in E.items():
        if num_layers in subset or subset is None:
            if log:
                ax.semilogy(N, error, label=f"$L={num_layers}$", **kwargs)
            else:
                ax.plot(N, error, label=f"$L={num_layers}$", **kwargs)
    ax.set_xticks(N)
    return ax


def depth_convergence(d):
    return {j: [d[i][j] for i in d] for j in list(d.values())[0]}


def width_convergence(d):
    return {i: [d[i][j] for j in list(d.values())[0]] for i in d}


def plot_curve_convergences(savename, figblock):
    # Analytic diffeomorphism
    g = LogStepDiff()

    # Define Curves
    c1 = Infinity()
    c0 = ComposedCurve(c1, g)

    num_layers_list = list(range(1, 25))
    num_functions_list = list(range(1, 25))
    subset = [1, 3, 5, 7, 10, 15, 23, 24]

    d = create_convergence_dict(
        c0,
        c1,
        num_layers_list,
        num_functions_list,
        curve_reparametrization,
        transform="qmap",
        logger=Silent(),
        max_iter=500,
        k=1024,
        device="cuda"
    )
    with open("curve-converge.json", "w") as f:
        json.dump(d, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.set_prop_cycle(MONOCHROME)
    plot_depth_convergence(d, ax1, subset=subset)
    ax1.legend()
    ax1.set_xlabel("$L$", fontsize=18)
    ax1.set_ylim(1e-7, 10)
    ax1.set_xticks(num_layers_list[::2], num_layers_list[::2])

    ax2.set_prop_cycle(MONOCHROME)
    plot_width_convergence(d, ax2, subset=subset)
    ax2.legend(loc=3)
    ax2.set_xlabel("$M$", fontsize=18)
    ax2.set_ylim(1e-7, 10)
    ax2.set_xticks(num_functions_list[::2], num_functions_list[::2])
    savefig(savename)
    savefig(savename, suffix=".eps")
    if figblock:
        plt.show(block=figblock)


def plot_surface_convergences(savename, figblock):
    g = RotationDiffeomorphism()
    f1 = HyperbolicParaboloid()
    f0 = f1.compose(g)
    num_layers_list = list(range(1, 16))
    num_functions_list = list(range(1, 16))
    subset = [1, 3, 5, 7, 10, 15]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    d = create_convergence_dict(
        f0,
        f1,
        num_layers_list,
        num_functions_list,
        surface_reparametrization,
        transform="qmap",
        k=32,
        logger=Silent(),
        device=device,
    )
    with open("surface-converge.json", "w") as f:
        json.dump(d, f)

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 3))
    ax3.set_prop_cycle(MONOCHROME)
    plot_depth_convergence(d, ax3, subset=subset, label_identifier="N")
    ax3.legend(loc="lower left")
    ax3.set_xlabel("$L$", fontsize=18)

    ax4.set_prop_cycle(MONOCHROME)
    plot_width_convergence(d, ax4, subset=subset)
    ax4.legend(loc=3)
    ax4.set_xlabel("$N$", fontsize=18)
    savefig(savename)
    savefig(savename, suffix=".eps")
    if figblock:
        plt.show(block=figblock)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    print()
    plot_curve_convergences("Fig10.png", args.show)
    plot_surface_convergences("Fig11.png", args.show)
