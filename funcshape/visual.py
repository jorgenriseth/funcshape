from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from funcshape.utils import col_linspace
from funcshape.diffeomorphism import Diffeomorphism1D


def create_figsaver(figpath):
    figpath = Path(figpath)
    figpath.mkdir(exist_ok=True, parents=True)
    def savefig(name, fig=None):
        path = figpath / name
        if fig is None:
            plt.savefig(figpath / name, bbox_inches="tight")
        else:
            fig.savefig(figpath / name, bbox_inches="tight")
    return savefig


def plot_curve(c, npoints=201, dotpoints=None, ax=None, **kwargs):
    X = torch.linspace(0, 1, npoints).unsqueeze(-1)
    C = c(X)
    cx, cy = C[:, 0], C[:, 1]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(cx, cy, **kwargs)

    if dotpoints is not None:
        X = torch.linspace(0, 1, dotpoints).unsqueeze(-1)
        C = c(X)
        cx, cy = C[:, 0], C[:, 1]
        ax.plot(cx, cy, c=ax.lines[-1].get_color(), ls='none', marker='o', markeredgecolor="black")
    return ax
    

def get_plot_data(q, r, network, npoints):
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)
    with torch.no_grad():
        y = network(x)
        u = network.derivative(x)
        Q, R = q(x), torch.sqrt(u) * r(y)
    return x, y, u, Q, R


def plot_diffeomorphism_1d(f, npoints=201, ax=None, **kwargs):
    with torch.no_grad():
        X = col_linspace(0, 1, npoints)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(X, f(X), **kwargs)
        return ax


def plot_derivative(f, npoints=401, ax=None, **kwargs):
    return plot_diffeomorphism_1d(lambda x: f.derivative(x, h=1e-4), npoints, ax,
                               **kwargs)


def get_plot_data(f, k=32):
    K = k**2
    X = torch.rand(K, 2)

    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)

    X = torch.cat((X, Y), dim=1)

    Z = f(X).detach().numpy().T
    return Z.reshape(-1, k, k)


def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_diffeomorphism_2d(f, k=16, ax=None, **kwargs):
    K = k**2
    X = torch.rand(K, 2)
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    X = torch.cat((X, Y), dim=1)
    Z = f(X).reshape(k, k, 2).detach()
    plot_grid(Z[:, :, 0], Z[:, :, 1], ax=ax, **kwargs)



def plot_surface(f, ax=None, colornorm=None, k=32, camera=(30, -60), **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    
    coloring = get_plot_data(f.volume_factor, k=k).squeeze()
    if colornorm is None:
        colors = coloring / coloring.max()
    else:
        colors = colornorm(coloring)
        
    Z = get_plot_data(f, k=k)
    
    ax.plot_surface(*Z, shade=False, facecolors=cm.jet(colors), rstride=1, cstride=1, **kwargs)
    ax.view_init(*camera)
    return ax


def get_common_colornorm(surfaces, k=128):
    colors = [get_plot_data(fi.volume_factor, k=k).squeeze() for fi in surfaces]
    return Normalize(vmin=min([ci.min() for ci in colors]), vmax=max([ci.max() for ci in colors]))
        
        
# Create cycler object for gray-scale figures. Use any styling from above you please
MONOCHROME = (
    cycler('color', ['k']) * 
    cycler('marker', ['', 'd', '^', '.']) * 
    cycler('linestyle', ['-', '--', ':', '-.'])
)