# FuncShape
[![DOI](https://zenodo.org/badge/514912162.svg)](https://zenodo.org/badge/latestdoi/514912162)

This repository contains code accompanying the paper "Deep learning of diffeomorphisms for optimal reparametrizations
of shapes" by Celledoni, Glöckner, Riseth, Schmeding. The figures from the paper may be seen and created from the notebooks:
1. [Curve Reparametrization](notebooks/curves-reparametrization.ipynb)
2. [Surface Reparametrization](notebooks/surfaces-reparametrization.ipynb)
3. [MNIST Digit Matching](notebooks/digit-matching.ipynb)
4. [Norm Estimates](notebooks/norm-estimates.ipynb)

---
## Setup
We only provide instructions for replicating the enironment using `conda`. This may be done by
```bash
conda env create -f environment.yml
```

To install the source code from `funcshape` as  a library:
```bash
pip install -e .
```
The '-e' is optional and allows you to change the library after installation. Mainly intended for people intending further development. 


### MNIST Data
The image data from the hand-written digits are the ones used in 

Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998, \cite{lecun-98}. 

The image data may be downloaded by running
```bash
./download-mnist.sh
```


## Citing
Please use the following citation if you are building upon our work,
```
[Paper here]
```

If you rather want to explicitly refer to something from the implementation, you may use

```
[Zenodo Code Here]
```

---
## Contents
The main part of the source code is defined within the `funcshape`  folder. The folder contains the following modules:
* **layers**
    - **layerbase.py**: Defines an abstract base-class to be extended with new layertypes.
    - **sineseries.py**: Defines a pytorch module/layer for reparametrization of *curves*, using sine series.
    - **sinefourier.py**: Defines a pytorch module/layer for reparametrization of *surfaces*, using basis functions created as a tensor product of sine-series in one direction, and a full Fourier series in the other.
* **testlib**:
    * **curves.py**: Library of parametric curves and one-dimensional diffeomorphisms.
    * **surfaces.py**: Library of parametric surfaces and two-dimensional diffeomorphisms.
* **curve.py**: Class for defining multidimensional curves with a few helper functions.
* **derivatives.py**: Contains numerical approximations of derivatives, jacobians and other related functions.
* **diffeomorphism.py**: Contains classes for defining diffeomorphisms for both curves and surfaces.
* **gradient_descent.py**: Implements the Riemannian gradient descent algorithm for reparametrization of curves.
* **imageinterp.py**: Implements functions for creating parametric surfaces from image data.
* **interpolation.py**: Implements functions for interpolating between different curves and surfaces.
* **logging.py**: Immplements a Logging-class which is used to adjust verbosity of error logging while training the network.
* **loss.py**: Implements loss functions for curves and surfaces.
* **networks.py**: Implements a pytorch-module to collect layers into a neural network.
* **reparametrize.py**: Implements the reparametrization algorithm. Defines a single interface, which optionally calls a function to be used with torch's LBFGS-optimizer.
* **surface.py**: Implements a surface class, with various helper functions.
* **transforms.py**: Implements curve/surface transforms such as the *SRVT* or *Qmap*.
* **utils.py**: Contains various helper functions.
* **visual.py**: Defines functions for easier plotting and visualization.
---
