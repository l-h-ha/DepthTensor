from typing import List, Set

from . import Tensor
from ._core.exceptions import (
    GradientComputationError, GRADIENT_COMPUTATION_ERROR, 
    CuPyNotFound, CUPY_NOT_FOUND_MSG
)

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

def differentiate(tensor: Tensor) -> List[Tensor]:
    topo: List[Tensor] = []
    visited: Set[Tensor] = set()

    def build(t: Tensor):
        if t in visited:
            return
        visited.add(t)
        
        for prev in t.prev:
            build(prev)
        topo.append(t)

    build(tensor)

    if tensor.device == "gpu":
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        tensor.grad = cp.ones(tensor.shape, tensor.dtype)
    elif tensor.device == "cpu":
        tensor.grad = np.ones(tensor.shape, tensor.dtype)

    for t in reversed(topo):
        if t.backward is None:
            if not t.requires_grad:
                raise GradientComputationError(f"Attempted to propagate backward off of a tensor ({t}), which is not differentiable.")
            if len(t.prev) > 0:
                raise GradientComputationError(f"Tensor ({t})'s backward function is None.")
            continue #* t has no preceeding operations, so t is a leaf node, which cannot be propagated further.
        t.backward()
    return topo