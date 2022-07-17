import torch
from torch import sin, cos
from torch.nn import Parameter, Upsample
from numpy import pi

from funcshape.derivatives import jacobian
from funcshape.layers.layerbase import SurfaceLayer


class SineFourierLayer(SurfaceLayer):
    def __init__(self, n, init_scale=0.0, p=1):
        super().__init__()

        # Number related to basis size
        self.n = n
        self.N = 2 * n**2 + n

        # Upsampler required in forward pass
        self.upsample = Upsample(scale_factor=n, mode='nearest')

        # Create weight vector
        self.weights = Parameter(
            init_scale * torch.randn(2*self.N, requires_grad=True)
        )

        # Vectors used function evaluation and projection.
        self.nvec = torch.arange(1, n+1, dtype=torch.float)
        self.L = self.lipschitz_vector()
        self.p = p
        self.project()
        self.I = torch.eye(2, 2)

    def to(self, device):
        super().to(device)
        self.device = device
        self.nvec = self.nvec.to(device)
        self.L = self.L.to(device)
        self.I = self.I.to(device)
        return self

    def forward(self, x):
        """Assumes input on form (K, 2)"""
        # Possible alternative: K = np.prod(x.shape[:-1])
        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)

        # Sine matrices
        S1 = sin(pi * z) / (self.nvec * pi)
        S2 = sin(2 * pi * z)[:, (1, 0), :] / (self.nvec)

        # Cosine matrices
        C2 = cos(2 * pi * z)[:, (1, 0), :] / (self.nvec)

        # Tensor product matrices.
        T2 = self.upsample(S1) * S2.repeat(1, 1, n)
        T3 = self.upsample(S1) * C2.repeat(1, 1, n)

        # Vector field evaluation.
        B = torch.zeros(K, 2, 2*N, device=x.device)

        B[:, 0, :n] = S1[:, 0, :]  # Type 1 x direction
        B[:, 1, N:(N+n)] = S1[:, 1, :]  # Type 1  y-direction

        B[:, 0, n:(n**2+n)] = T2[:, 0, :]  # Type 2 x-direction
        B[:, 1, (N+n):(N + n**2 + n)] = T2[:, 1, :]  # Type 2 y-direction

        B[:, 0, (n+n**2):N] = T3[:, 0, :]  # Type 3 x-direction
        B[:, 1, (N+n+n**2):] = T3[:, 1, :]  # Type3 y-direction

        return x + B @ self.weights

    def derivative(self, x, h=None):
        """Assumes input on form (K, 2)"""
        if h is not None:
            return jacobian(self, x, h)

        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)

        # Sine matrices
        S1 = sin(pi * z)  / self.nvec
        S2 = sin(2 * pi * z)[:, (1, 0), :]

        # Cosine matrices
        C1 = cos(pi * z)
        C2 = cos(2 * pi * z)[:, (1, 0), :]

        # Now for derivative matrices
        T11 = self.upsample(C1) * ( (S2 / self.nvec)).repeat(1, 1, n)
        T12 = self.upsample(S1) * C2.repeat(1, 1, n)

        T21 = self.upsample(C1) * ((C2 / self.nvec)).repeat(1, 1, n)
        T22 = self.upsample(S1) * (-S2).repeat(1, 1, n)

        # Create and fill a tensor with derivative outputs
        D = torch.zeros(K, 2, 2, 2*N, device=x.device)

        D[:, 0, 0, :n] = C1[:, 0, :]  # Type 1 x direction dx
        D[:, 1, 1, N:(N+n)] = C1[:, 1, :]  # Type 1  y-direction dy

        D[:, 0, 0, n:(n + n**2)] = T11[:, 0, :]  # Type 2 x-direction dx
        D[:, 0, 1, n:(n + n**2)] = T12[:, 0, :]  # Type 2 x-direction dy
        D[:, 1, 1, (N+n):(N + n + n**2)] = T11[:, 1, :]  # Type 2 y-direction dy
        D[:, 1, 0, (N+n):(N + n + n**2)] = T12[:, 1, :]  # Type 2 x-direction dy

        D[:, 0, 0, (n+n**2):N] = T21[:, 0, :]  # Type 3 x-direction dx
        D[:, 0, 1, (n+n**2):N] = T22[:, 0, :]  # Type 3 x-direction dy

        D[:, 1, 1, (N+n+n**2):] = T21[:, 1, :]  # Type 3 y-direction dy
        D[:, 1, 0, (N+n+n**2):] = T22[:, 1, :]  # Type 3 y-direction dx

        return self.I + D @ self.weights

    def lipschitz_vector(self):
        n, N = self.n, self.N
        upsampled = self.upsample(self.nvec.view(1, 1, -1)).squeeze()
        repeated = self.nvec.repeat(n)
        T23 = (torch.sqrt(4 * upsampled**2 + repeated**2) /
               (2 * upsampled * repeated)).repeat(2)
        # T23 = (1. / repeated).repeat(2)

        Li = torch.ones(2*N)
        Li[n:N] = T23
        Li[(N+n):] = T23
        return Li

    def project(self, method: str = "lipschitz", **kwargs):
        with torch.no_grad():
            L = (torch.abs(self.weights * self.L)).sum()  # + self.eps

            if L >= 1. - 1e-6:
                self.weights *= (1 - 1e-6) / L