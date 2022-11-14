from pathlib import Path

import matplotlib.pyplot as plt
import torch
import idx2numpy 
from torch import optim
from torchvision.transforms.functional import gaussian_blur

from funcshape.transforms import Qmap2D
from funcshape.imageinterp import SingleChannelImageSurface
from funcshape.layers.sinefourier import SineFourierLayer
from funcshape.interpolation import linear_interpolate
from funcshape.networks import SurfaceReparametrizer
from funcshape.loss import ImageComponentDistance
from funcshape.reparametrize import reparametrize
from funcshape.logging import Logger
from funcshape.utils import torch_square_grid
from savefig import savefig

def interpolate_digits(savename, figblock=True):
    # Load Data
    imgfile = str(Path(__file__).parent / "../data/t10k-images-idx3-ubyte")
    imgdata = torch.tensor(idx2numpy.convert_from_file(imgfile), dtype=torch.float )
    imgdata = gaussian_blur(imgdata, [7,7])
    imgdata /= imgdata.max()

    # Extract example images
    img1 = imgdata[1]  # Switch index here to test with various digits
    img2 = imgdata[17]

    # Create surface representation of images
    f = SingleChannelImageSurface(img1, centering=False, scaling=False, mode='bicubic')
    g = SingleChannelImageSurface(img2, centering=False, scaling=False, mode='bicubic')

    # Create sample data of Q-map representations of the images.
    q = Qmap2D(f)
    r = Qmap2D(g)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Define reparametrization-network
    RN = SurfaceReparametrizer(
        [SineFourierLayer(4) for i in range(10)]
    ).to(device)

    optimizer = optim.LBFGS(RN.parameters(), max_iter=200, line_search_fn="strong_wolfe")
    loss_func = ImageComponentDistance(q, r, k=32, h=1e-4).to(device)
    errors = reparametrize(RN, loss_func, optimizer, 200, Logger(1))
    RN.detach()
    RN.to("cpu"), loss_func.to("cpu")

    # Find example data of surface representations of the images.
    k = 64
    XX = torch_square_grid(k=k)
    X = XX.view(-1, 2)
    # Plot surfaces as images
    numsteps = 7
    fig = plt.figure(figsize=(21, 6))
    for i, h in enumerate(linear_interpolate(g, f, numsteps)):
        ax = fig.add_subplot(2, numsteps, i+1)
        Z = h(X).view(k, k, 3).permute(2, 0, 1).detach().numpy()
        ax.contourf(*Z)# optimizer = optim.SGD(RN.parameters(), lr=1e-1, momentum=0.2)

        ax.invert_yaxis()
        ax.set_axis_off()

    for i, h in enumerate(linear_interpolate(lambda x: g(RN(x)), f, numsteps)):
        ax = fig.add_subplot(2, numsteps, i+(numsteps+1))
        Z = h(X).view(k, k, 3).permute(2, 0, 1).detach().numpy()
        ax.contourf(*Z)
        ax.invert_yaxis()
        ax.set_axis_off()

    savefig(savename)
    plt.show(block=figblock)

if __name__ == "__main__":
    interpolate_digits("Fig9.png", False)