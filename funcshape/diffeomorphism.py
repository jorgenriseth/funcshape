import torch

from funcshape.derivatives import central_differences, batch_determinant


class Diffeomorphism1D:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def derivative(self, x, h=5e-4):
        return central_differences(self, x, h)

    def compose(self, f):
        return Diffeomorphism1D(lambda x: self(f(x)))


class Diffeomorphism2D:
    def __init__(self, component_function_tuple):
        super().__init__()
        self.S = tuple(component_function_tuple)

    def __call__(self, X):
        out = torch.zeros_like(X)
        out[..., 0] = self.S[0](X)
        out[..., 1] = self.S[1](X)
        return out

    def partial_derivative(self, X, component, h):
        H = torch.zeros_like(X, device=X.device)
        H[..., component] = h
        return (0.5 / h) * torch.cat([(ci(X + H) - ci(X - H)).unsqueeze(dim=-1) for ci in self.S], dim=-1)

    def jacobian(self, X, h):
        dxf = self.partial_derivative(X, 0, h)
        dyf = self.partial_derivative(X, 1, h)
        return torch.stack((dxf, dyf), dim=-1)
    
    def jacobian_determinant(self, X, h):
        D = self.jacobian(X, h)
        return batch_determinant(D)

    def compose(self, f):
        return Diffeomorphism2D((
            lambda x: self.S[0](f(x)),
            lambda x: self.S[1](f(x)),
        ))
