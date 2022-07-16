import torch


def central_differences(f, x, h=1e-4, **kwargs):
    return (0.5 / h) * (f(x + h, **kwargs) - f(x - h, **kwargs))


def partial_differences(f, x, component, h=1e-4):
    dx = torch.zeros_like(x)
    dx[..., component] = h
    return (0.5 / h) * (f(x + dx) - f(x - dx))


def jacobian(f, x, h=1e-4):
    dxf = partial_differences(f, x, 0, h=1e-4)
    dyf = partial_differences(f, x, 1, h=1e-4)
    return torch.stack((dxf, dyf), dim=-1)


def batch_determinant(B):
    """ Compute determinant of a batch of 2x2 matrices. """
    assert B.dim() == 3, f"Dim.shape should be (K, 2, 2), got {B.shape}"
    return (B[:, 0, 0] * B[:, 1, 1] - B[:, 1, 0] * B[:, 0, 1]).view(-1, 1)


def batch_trace(B):
    """ Compute trace of a batch of 2x2 matrices. """
    assert B.dim() == 3, f"Dim.shape should be (K, 2, 2), got {B.shape}"
    return (B[:, 0, 0] + B[:, 1, 1]).view(-1, 1)
