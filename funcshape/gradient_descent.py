import numpy as np
import torch
from numpy import pi

from funcshape.derivatives import central_differences
from funcshape.utils import col_linspace

def sqnorm(x):
    return (x**2).sum()


def error_func(Q, r, y, u):
    return 0.5 * (((Q - u * r(y))**2).sum() - 0.5 * ((Q[0]-u[0]*r(y[0]))**2 + (Q[-1]-u[-1]*r(y[-1]))**2).sum()) / (Q.shape[0]-1)

def eval_palais(x, N):
    z = N * x  # * 2pi, but this is computed in the outer gradient_descent function
    V1 = torch.sin(z) / (np.sqrt(2) * pi * N)
    V2 = (1. - torch.cos(z)) / (np.sqrt(2) * pi * N)
    return torch.cat((V1, V2), dim=-1)


def get_frequency_vector(n, dim):
    return 2 * pi * torch.arange(1, n+1).reshape(-1, *[1 for _ in range(dim)])


def get_bound(c):
    return 1. / (np.sqrt(2) * c.norm(1))


def gradient_descent(q, r, k, n, h=1e-4, max_iter=1000, backtrack_c=0.1, rho=0.9, gtol=1e-8,
                     rtol=1e-8, verbose=False):
    x = col_linspace(0, 1, k)
    Q = q(x)
    dQ = central_differences(q, x, h=h)
    y = col_linspace(0, 1, k)
    u = torch.ones_like(y)

    # Compute vector of (2 * pi * m) for m = 1, ..., n
    N = 2. * pi * torch.arange(1, n+1)

    # Init convergence vector, and prepare for iteration.
    error = np.nan * np.empty(max_iter+1)
    error[0] = error_func(Q, r, y, u)
    deltaE = np.inf

    if verbose:
        print(f"{'Iter':5}\t{'E':10}\t{'|c|':10}\t{'α':10}\t{'δEi/E0':10}")
        print(f"{0:5d}\t{error[0]:10.4e}\t {' ':10}\t{' ':10}\t")
    i = 1

    while deltaE / error[0] > rtol and i <= max_iter:
        # Reevaluate r and dr at points y_n = y_{n-1} - ε * Σ ci * vi(y_{n-1}) = φ_n(y_{n-1})
        R = r(y)
        dR = central_differences(r, y, h=h) * u
        DE = (dQ * R - Q * dR).sum(dim=-1)

        # Evaluate basis and derivatives.
        V = eval_palais(y, N)
        dV = central_differences(eval_palais, y, h=h, N=N)
        
        # Find coefficients
        c = (DE @ V) / (k-1)

        grad_norm = c.norm()  # Orthonormal basis, simplifies gradient computation
        if grad_norm < gtol:
            print(f"|c| = {grad_norm} < {gtol}. Exiting")
            break

        # Compute discrete projected gradient dEj = Σ_i <DE, vi> vi(xj)
        dE = (V @ c).view(-1, 1)
        dEdx = (dV @ c).view(-1, 1)

        # Find stepsize by armijo backtracking
        step = get_bound(c)
        step = backtracking_linesearch(Q, r, y, u, step, dE, dEdx, c=backtrack_c, rho=rho)

        # Update points
        y = y - step * dE
        u = u * (1. - step * dEdx)
        error[i] = error_func(Q, r, y, u)
        deltaE = error[i-1] - error[i]
        i += 1

        if verbose:
            print(
                f"{i-1:5d}\t{error[i-1]:10.6e}\t{grad_norm:10.6e}\t{step:10.6e}\t{deltaE / error[0]:10.6e}"
            )

    return y, error[~np.isnan(error)]


def make_phi(Q, r, y, u, dy, du):
    def phi(a):
        return error_func(Q, r, y - a*dy, u*(1. - a * du))
    return phi


def backtracking_linesearch(Q, r, y, u, a0, dy, du, c=0.1, rho=0.1, max_iter=1000, verbose=False):
    a = a0
    phi = make_phi(Q, r, y, u, dy, du)
    phi0 = phi(0.)
    dphi0 = central_differences(phi, 0.)

    if dphi0 >= 0.:
        return 0.

    it = 0
    while (phi(a) >= phi0 + c * a * dphi0) and (it < max_iter):
        a = rho * a
        it += 1
        if verbose:
            print(
                f"[Backtrack]: {phi(a):>10.8e}\t {phi0 + c * a * dphi0:>10.8e}"
            )

    if(it == max_iter):
        print(
            f"[Backtrack]: phi0: {phi0:>10.3g}\t dphi0: {dphi0:>10.3g}\t a0: {a0:>10.3g}\t a: {a:>10.3g}"
        )

    return a
