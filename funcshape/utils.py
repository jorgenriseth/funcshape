import numpy as np
import torch

def numpy_nans(dim, *args, **kwargs):
    arr = np.empty(dim, *args, **kwargs)
    arr.fill(np.nan)
    return arr


def col_linspace(start, end, N):
    return torch.linspace(start, end, N).unsqueeze(-1)


def torch_square_grid(k=64):
    Y, X = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X = torch.stack((X, Y), dim=-1)
    return X