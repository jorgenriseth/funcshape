import numpy as np


def interpolate(f, g, t):
    return lambda x: (1.0 - t) * f(x) + t * g(x)


def linear_interpolate(f, g, steps):
    t = np.linspace(0, 1, steps)
    return [interpolate(f, g, ti) for ti in t]


def inverse_srvt(q):
    def inv(x):
        Q = q(x)
        h = 1.0 / (x.shape[0] - 1)
        points = Q * Q.norm(dim=-1, keepdim=True)
        return h * points.cumsum(dim=0)

    return inv


def srvt_interpolate(q, r, t):
    new_q = interpolate(q, r, t)
    return inverse_srvt(new_q)


def geodesic(q, r, steps):
    t = np.linspace(0, 1, steps)
    return [srvt_interpolate(q, r, ti) for ti in t]
