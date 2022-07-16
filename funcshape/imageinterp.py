import torch
import numpy as np
import torch.nn as nn
from funcshape.surface import Surface


class ImageInterpolator:
    def __init__(self, img, mode="bilinear", **kwargs):
        """ Assumes img of shape (C, H, W)"""
        if img.dim() == 2:
            self.img = img.view(1, 1, *img.shape).clone().detach()
            self.H, self.W = img.shape
            self.C = 1
        elif img.dim() == 3:
            self.img = img.view(1, *img.shape).clone().detach()
            self.C, self.H, self.W = img.shape
        else:
            raise ValueError(
                f"img should be of shape (C, H, W) or (H, W), got {img.shape}")

        self.mode = mode

    def __call__(self, x):
        if x.dim() == 2:
            return self.eval_point_list(x)
        elif x.dim() == 3:
            return self.eval_grid(x)
        raise ValueError(
            f"Got x with shape {x.shape}, should be (N, 2) or (H, W, 2)")

    def eval_point_list(self, x):
        """ Assumes the input comes on the form (N, 2), i.e. a list of points  
        points (x, y) placed within the domain D = [0, 1]^2"""
        npoints = x.shape[0]
        H = int(np.sqrt(npoints))
        self.img = self.img.to(x.device)

        #  Map input from [0, 1] -> [-1, 1] (required by grid_sample)
        X = x.view(1, H, H, 2)
        X = X * 2. - 1.

        out = nn.functional.grid_sample(self.img, X, mode=self.mode,
                                        align_corners=True, padding_mode="border")
        # Reshape output (C, H, H) -> (H * H, C)
        return out.view(self.C, npoints).transpose(1, 0).squeeze()

    def eval_grid(self, x):
        """ Assumes the input comes on the form (H, W, 2), i.e. a grid of points  
        points (x, y) placed within the domain D = [0, 1]^2"""

        #  Map input from [0, 1] -> [-1, 1] (required by grid_sample)
        X = x.view(1, *x.shape)
        X = X * 2. - 1.

        # Interpolate, and reshape output to input-form
        out = nn.functional.grid_sample(self.img, X, mode=self.mode,
                                        align_corners=True, padding_mode="border")

        return out.view(self.C, *x[..., 0].shape).permute(1, 2, 0).squeeze()


class SingleChannelImageSurface(Surface):
    def __init__(self, img, centering=True, scaling=False, **kwargs):
        if centering:
            self.center_x, self.center_y = self.find_center(img)
        else:
            self.center_x = 0.
            self.center_y = 0.

        if scaling:
            self.scale = image_area(img)
        else:
            self.scale = 1.

        self.img = ImageInterpolator(img)

        super().__init__((
            lambda x: x[..., 0] - self.center_x,
            lambda x: x[..., 1] - self.center_y,
            lambda x: self.img(x) / self.scale
        ))

    def find_center(self, im):
        nx, ny = im.shape[-2:]
        mass = im.sum((-1, -2))
        sx = (im * torch.arange(nx)).sum((-1, -2)) / (mass * nx)
        sy = (im * torch.arange(ny)).sum((-1, -2)) / (mass * nx)
        return sx, sy


class MultiChannelImageSurface(Surface):
    def __init__(self, img, mode="bilinear", **kwargs):
        """ Assumes image input on form (C, H, W)"""
        super().__init__((
            ImageInterpolator(img[0], mode, **kwargs),
            ImageInterpolator(img[1], mode, **kwargs),
            ImageInterpolator(img[2], mode, **kwargs)
        ))


def eye_offset(N, k=0):
    return torch.diag(torch.ones(N-abs(k)), diagonal=k)


def finite_difference_matrix(N):
    D = eye_offset(N, k=1) - eye_offset(N, k=-1)
    D[0, :2] = torch.tensor((-2, 2))
    D[-1, -2:] = torch.tensor((-2, 2))
    return D


def trapezian_weight_vector(N):
    w = torch.ones(N, 1)
    w[[0, -1]] = 0.5
    return w


def image_area(im):
    N = im.shape[0]
    h = 1 / (N - 1)
    D = finite_difference_matrix(N)
    w = trapezian_weight_vector(N)

    fxh = (im @ D.T)
    fyh = (D @ im)
    
    return float(0.5 * h * w.T @ (np.sqrt(4 * h**2 + fxh**2 + fyh**2)) @ w)
