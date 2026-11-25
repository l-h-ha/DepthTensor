from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..typing import TensorData
from .exceptions import GradientComputationError
from .utils import (
    _backward_active,
    _cast_gradient_work,
    _check_value_dtype,
    _backward_compute_dtype,
    get_device,
    to_tensordata,
)

if TYPE_CHECKING:
    from ..tensor import Tensor

def _topological_order(root: Tensor) -> list[Tensor]:
    from ..tensor import Tensor

    order = []
    visited: set[Tensor] = set()
    visiting = {root}
    stack = [(root, iter(root.prev))]

    while stack:
        tensor, parents = stack[-1]
        for parent in parents:
            if not isinstance(parent, Tensor) or parent in visited:
                continue

            if parent in visiting:
                raise GradientComputationError("Computation graph contains a cycle.")

            visiting.add(parent)
            stack.append((parent, iter(parent.prev)))
            break
        else:
            stack.pop()
            visiting.remove(tensor)
            visited.add(tensor)
            order.append(tensor)

    return order

def _prepare_seed(root: Tensor, seed: Tensor | TensorData | None) -> TensorData | None:
    from ..tensor import Tensor

    if isinstance(seed, Tensor):
        seed = seed.data

    if seed is None:
        return None

    _check_value_dtype(seed)

    if root.device != get_device(seed):
        raise RuntimeError("Gradient device must match the tensor.")

    if root.shape != seed.shape:
        raise RuntimeError("Gradient shape must match the tensor.")

    seed = to_tensordata(seed, root.device)
    dtype = _backward_compute_dtype(seed.dtype, root.dtype)
    return _cast_gradient_work(seed, dtype, copy=True)

def _final_gradient(tensor: Tensor, previous: TensorData | None) -> TensorData | None:
    gradient = tensor.grad
    if gradient is None:
        return previous

    if previous is not None:
        dtype = _backward_compute_dtype(tensor.dtype, gradient.dtype, previous.dtype)
        gradient = _cast_gradient_work(gradient, dtype) + _cast_gradient_work(previous, dtype)

    with np.errstate(over="ignore", under="ignore"):
        return _cast_gradient_work(gradient, tensor.dtype)

def _backward(root: Tensor, seed: Tensor | TensorData | None) -> list[Tensor]:
    if not root.requires_grad:
        raise RuntimeError("Cannot run backward on an undifferentiable tensor.")

    _check_value_dtype(root.data)

    order = _topological_order(root)
    seed = _prepare_seed(root, seed)

    previous_gradients = [tensor.grad for tensor in order]
    token = _backward_active.set(True)
    try:
        for tensor in order:
            tensor.grad = None

        if seed is None:
            root.one_grad()
        else:
            root.grad = seed

        for tensor in reversed(order):
            if tensor.grad_fn is not None:
                tensor.grad_fn()
            elif tensor.requires_grad and tensor.prev:
                raise GradientComputationError("Missing backward function.")

        # Store the final gradients only after every backward function succeeds.
        gradients = [
            _final_gradient(tensor, previous if tensor is not root else None)
            for tensor, previous in zip(order, previous_gradients)
        ]

        for tensor, gradient in zip(order, gradients):
            tensor.grad = gradient
    except BaseException:
        for tensor, previous in zip(order, previous_gradients):
            tensor.grad = previous
        raise
    finally:
        _backward_active.reset(token)

    return order
