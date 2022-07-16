import torch


class Surface:
    """ Torch-compatible surface class. Constructed from a tuple of functions,
    mapping tensor of dim (..., 2) to R^len(tuple) """
    def __init__(self, component_function_tuple, **kwargs):
        super().__init__(**kwargs)
        self.S = tuple(component_function_tuple)
        self.dim = len(self.S)

    def __call__(self, X):
        return torch.cat([ci(X).unsqueeze(dim=-1) for ci in self.S], dim=-1)

    def partial_derivative(self, X, component, h=3.4e-4):
        H = torch.zeros_like(X)
        H[..., component] = h
        return 0.5 * torch.cat([(ci(X + H) - ci(X - H)).unsqueeze(dim=-1) for ci in self.S], dim=-1) / h

    def volume_factor(self, X, h=1e-4):
        return torch.norm(self.normal_vector(X, h), dim=-1, keepdim=True)

    def normal_vector(self, X, h=3.4e-4):
        dfx = self.partial_derivative(X, 0, h)
        dfy = self.partial_derivative(X, 1, h)
        return torch.cross(dfx, dfy, dim=-1)

    def compose(self, f):
        return Surface((
            lambda x: self.S[0](f(x)),
            lambda x: self.S[1](f(x)),
            lambda x: self.S[2](f(x))
        ))
