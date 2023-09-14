import torch

from funcshape.diffeomorphism import Diffeomorphism2D


class Surface:
    """Torch-compatible surface class. Constructed from a tuple of functions,
    mapping tensor of dim (..., 2) to R^len(tuple)"""

    def __init__(self, component_function_tuple, **kwargs):
        super().__init__(**kwargs)
        self.S = tuple(component_function_tuple)
        self.dim = len(self.S)

    def __call__(self, X):
        return torch.cat([ci(X).unsqueeze(dim=-1) for ci in self.S], dim=-1)

    def partial_derivative(self, X, component, h):
        if h is None:
            raise ValueError(
                f"{self.__class__} has not implemented partial"
                + f" derivatives. Needs variable h={h} to be float"
                + f" to enable finite difference approximation."
            )
        H = torch.zeros_like(X, device=X.device)
        H[..., component] = h
        return (0.5 / h) * torch.cat(
            [(ci(X + H) - ci(X - H)).unsqueeze(dim=-1) for ci in self.S], dim=-1
        )

    def volume_factor(self, X, h):
        return torch.norm(self.normal_vector(X, h), dim=-1, keepdim=True)

    def normal_vector(self, X, h):
        dfx = self.partial_derivative(X, 0, h)
        dfy = self.partial_derivative(X, 1, h)
        return torch.cross(dfx, dfy, dim=-1)

    def compose(self, f):
        return Surface(
            (
                lambda x: self.S[0](f(x)),
                lambda x: self.S[1](f(x)),
                lambda x: self.S[2](f(x)),
            )
        )


class ComposedSurface(Surface):
    def __init__(self, surf: Surface, diffeomorphism: Diffeomorphism2D):
        self.s = surf
        self.diffeo = diffeomorphism
        self.dim = surf.dim

    def __call__(self, X):
        return self.s(self.diffeo(X))

    def normal_vector(self, X, h):
        J = self.diffeo.jacobian_determinant(X, h)
        n = self.s.normal_vector(self.diffeo(X), h)
        return J * n