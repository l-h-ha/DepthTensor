from . import Tensor
from .typing import TensorData
from ._core.exceptions import (
    GradientComputationError,
    GRADIENT_COMPUTATION_ERROR,
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
)

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None


def _grad_check(grad: TensorData, tensor: Tensor) -> None:
    if cp is not None and (tensor.device == "gpu" and not isinstance(grad, cp.ndarray)):
        raise RuntimeError(f"Expected gradient to be a cupy.ndarray, got: {type(grad)}")
    if tensor.device == "cpu" and not isinstance(grad, np.ndarray):
        raise RuntimeError(
            f"Expected gradient to be a numpy.ndarray, got: {type(grad)}"
        )
    if grad.shape != tensor.shape:
        raise RuntimeError(
            f"Gradient's shape is incompatible with tensor's. Expected {tensor.shape}, got {grad.shape}"
        )


def differentiate(tensor: Tensor, grad: TensorData | None = None) -> list[Tensor]:
    """
    Compute the gradients of the tensor with respect to its ancestors.

    This function performs reverse-mode automatic differentiation (backpropagation).
    It traverses the computation graph backwards from the given tensor and
    computes the gradients for all tensors that require gradients.

    Parameters
    ----------
    tensor : Tensor
        The tensor to start the backward pass from.
    grad : TensorData | None, optional
        The gradient to seed the backward pass with. If None, it defaults to
        ones with the same shape as `tensor`.

    Returns
    -------
    list[Tensor]
        A list of tensors in the computation graph in topological order (reversed).

    Raises
    ------
    RuntimeError
        If there is a mismatch in gradient device or type.
    GradientComputationError
        If a tensor in the graph is missing a backward function but requires gradients.
    CuPyNotFound
        If the device is 'gpu' and CuPy is not available.
    """
    topo: list[Tensor] = []
    visited: set[Tensor] = set()

    def build(t: Tensor):
        if t in visited:
            return
        visited.add(t)

        for prev in t.prev:
            if isinstance(prev, Tensor):
                build(prev)
        topo.append(t)

    build(tensor)

    if tensor.grad is None:
        if grad is not None:
            _grad_check(grad, tensor)
            tensor.grad = grad
        else:
            if tensor.device == "gpu":
                if cp is None:
                    raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                tensor.grad = cp.ones(tensor.shape, tensor.dtype)
            elif tensor.device == "cpu":
                tensor.grad = np.ones(tensor.shape, tensor.dtype)
    else:
        if grad is not None:
            _grad_check(grad, tensor)
            tensor.grad += grad
        else:
            if tensor.device == "gpu":
                if cp is None:
                    raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                tensor.grad = cp.ones(tensor.shape, tensor.dtype)
            elif tensor.device == "cpu":
                tensor.grad = np.ones(tensor.shape, tensor.dtype)

    for t in reversed(topo):
        if t.backward is None:
            if not t.requires_grad:
                continue
            if len(t.prev) > 0:
                raise GradientComputationError(
                    f"Tensor ({t})'s backward function is None."
                )
            continue
        t.backward()
    return topo
