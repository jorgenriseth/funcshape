import torch
import numpy as np
from funcshape.utils import numpy_nans


def reparametrize(
    network, loss, optimizer, iterations, logger, scheduler=None, projection_kwargs=None
):
    if projection_kwargs is None:
        projection_kwargs = {}

    if isinstance(optimizer, torch.optim.LBFGS):
        return reparametrize_lbfgs(
            network, loss, optimizer, logger, scheduler, projection_kwargs
        )

    # Evaluate initial error
    logger.start()
    error = numpy_nans(iterations + 1)
    error[0] = float(loss(network))

    for i in range(iterations):
        # Zero gradient buffers
        optimizer.zero_grad()

        # Compute current loss and gradients
        l = loss(network)
        l.backward()

        # Update optimizer if using scheduler
        if scheduler is not None:
            scheduler.step(l)

        # Update parameters
        optimizer.step()
        network.project(**projection_kwargs)

        error[i + 1] = loss.get_last()
        logger.log(it=i, value=error[i + 1])

    logger.stop()
    return error


def reparametrize_lbfgs(
    network, loss, optimizer, logger, scheduler=None, projection_kwargs=None
):
    if projection_kwargs is None:
        projection_kwargs = {}

    # Get max iterations from optimizer
    iterations = optimizer.defaults["max_eval"]

    # Evaluate initial error
    logger.start()

    global func_evals
    func_evals = 0

    global error
    error = numpy_nans(iterations + 2)
    # error[0] = float(loss(network))
    it = [0]

    def closure():
        global error
        global func_evals

        # Only log error after finishing line search
        if optimizer.state[optimizer._params[0]]["func_evals"] > func_evals:
            func_evals = optimizer.state[optimizer._params[0]]["func_evals"]
            logger.log(it=it[0], value=loss.get_last())
            it[0] += 1
            error[it[0]] = loss.get_last()

        # Set gradient buffers to zero.
        optimizer.zero_grad()

        # Compute loss, and perform a backward pass and gradient step
        network.project(**projection_kwargs)
        l = loss(network)
        l.backward()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(l)

        return l

    optimizer.step(closure)
    network.project(**projection_kwargs)
    logger.log(it=it[0], value=loss.get_last())
    error[it[0]] = loss.get_last()
    logger.stop()

    return error[~np.isnan(error)]
