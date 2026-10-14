"""
This module contains functions to perform power iteration with deflation
to compute the top eigenvalues and eigenvectors of a linear operator
"""
from typing import Optional, Tuple, Union

import numpy as np
import torch

from hessian_eigenthings.operator import Operator, LambdaOperator
import hessian_eigenthings.utils as utils


def deflated_power_iteration(
    operator: Operator,
    num_eigenthings: int = 10,
    power_iter_steps: int = 20,
    power_iter_err_threshold: float = 1e-4,
    momentum: float = 0.0,
    use_gpu: bool = True,
    fp16: bool = False,
    to_numpy: bool = True,
    return_rel_residuals: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute top k eigenvalues by repeatedly subtracting out dyads
    operator: linear operator that gives us access to matrix vector product
    num_eigenvals number of eigenvalues to compute
    power_iter_steps: number of steps per run of power iteration
    power_iter_err_threshold: early stopping threshold for power iteration
    return_rel_residuals: if True, also return the final relative residual
        ||Av - lambda v|| / ||Av|| of each eigenpair
    returns: np.ndarray of top eigenvalues, np.ndarray of top eigenvectors, and
        optionally np.ndarray of relative residuals
    """
    eigenvals = []
    eigenvecs = []
    rel_residuals = []
    current_op = operator
    prev_vec = None

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec

    utils.log("beginning deflated power iteration")
    for i in range(num_eigenthings):
        utils.log("computing eigenvalue/vector %d of %d" % (i + 1, num_eigenthings))
        eigenval, eigenvec, rel_residual = power_iteration(
            current_op,
            power_iter_steps,
            power_iter_err_threshold,
            momentum=momentum,
            use_gpu=use_gpu,
            fp16=fp16,
            init_vec=prev_vec,
            return_rel_residual=True,
        )
        utils.log("eigenvalue %d: %.4f" % (i + 1, eigenval))

        def _new_op_fn(x, op=current_op, val=eigenval, vec=eigenvec):
            return utils.maybe_fp16(op.apply(x), fp16) - _deflate(x, val, vec)

        current_op = LambdaOperator(_new_op_fn, operator.size)
        prev_vec = eigenvec
        eigenvals.append(eigenval)
        rel_residuals.append(rel_residual)
        eigenvec = eigenvec.cpu()
        if to_numpy:
            # Clone so that power_iteration can continue to use torch.
            numpy_eigenvec = eigenvec.detach().clone().numpy()
            eigenvecs.append(numpy_eigenvec)
        else:
            eigenvecs.append(eigenvec)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)
    rel_residuals = np.array(rel_residuals)

    # sort them in descending order
    sorted_inds = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_inds][::-1]
    eigenvecs = eigenvecs[sorted_inds][::-1]
    rel_residuals = rel_residuals[sorted_inds][::-1]
    if return_rel_residuals:
        return eigenvals, eigenvecs, rel_residuals
    return eigenvals, eigenvecs


def power_iteration(
    operator: Operator,
    steps: int = 20,
    error_threshold: float = 1e-4,
    momentum: float = 0.0,
    use_gpu: bool = True,
    fp16: bool = False,
    init_vec: Optional[torch.Tensor] = None,
    return_rel_residual: bool = False,
) -> Union[Tuple[float, torch.Tensor], Tuple[float, torch.Tensor, float]]:
    """
    Compute dominant eigenvalue/eigenvector of a matrix
    operator: linear Operator giving us matrix-vector product access
    steps: number of update steps to take
    return_rel_residual: if True, also return the relative residual
        ||Av - lambda v|| / ||Av|| of the final step, where v is the unit vector
        the operator was applied to and lambda = v . Av
    returns: (principal eigenvalue, principal eigenvector) pair, and optionally
        the relative residual
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    vector_size = operator.size  # input dimension of operator
    if init_vec is None:
        vec = torch.rand(vector_size)
    else:
        vec = init_vec

    vec = utils.maybe_fp16(vec, fp16)

    if use_gpu:
        vec = vec.cuda()

    prev_lambda = 0.0
    rel_residual = 1.0
    for i in range(steps):
        prev_vec = vec / (torch.linalg.vector_norm(vec) + 1e-6)
        new_vec = utils.maybe_fp16(operator.apply(vec), fp16) - momentum * prev_vec
        new_vec_norm = torch.linalg.vector_norm(new_vec)
        # need to handle case where we end up in the nullspace of the operator.
        # in this case, we are done.
        if new_vec_norm.item() == 0.0:
            if return_rel_residual:
                return 0.0, new_vec, 0.0
            return 0.0, new_vec
        lambda_estimate = prev_vec.dot(new_vec)
        # Relative residual of the unit vector the operator was applied to
        residual = new_vec - lambda_estimate * prev_vec
        rel_residual = torch.linalg.vector_norm(residual) / new_vec_norm
        diff = lambda_estimate - prev_lambda
        vec = new_vec.detach() / new_vec_norm
        # error is 1.0 when lambda_estimate == 0.0 (for low-rank)
        error = torch.where(
            lambda_estimate == 0.0,
            torch.ones_like(lambda_estimate),
            torch.abs(diff / lambda_estimate),
        )
        error = float(error)
        utils.progress_bar(i, steps, "power iter error: %.4f" % error)
        if error < error_threshold:
            break
        prev_lambda = lambda_estimate
    lambda_estimate = float(lambda_estimate)
    if return_rel_residual:
        return lambda_estimate, vec, float(rel_residual)
    return lambda_estimate, vec
