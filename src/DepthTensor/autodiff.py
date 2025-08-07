from typing import Union, List, TypeAlias, Optional, Any

from . import CPUtensor, GPUTensor
Tensor: TypeAlias = Union[CPUtensor, GPUTensor]

import numpy as np
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def differentiate(tensor: Tensor, upstream_grad: Optional[Union[Any, Tensor, np.ndarray]] = None) -> List[Tensor]:
    topo: List[Tensor] = []
    visited = set()

    def build(t: Tensor) -> None:
        if t is visited:
            return
        visited.add(t)
        for prev in t.prev:
            build(prev)
        topo.append(t)
    build(tensor)
    topo.reverse()

    if upstream_grad is not None:
        if isinstance(upstream_grad, Tensor):
            if not tensor.is_device(upstream_grad.device):
                raise RuntimeError("Upstream gradient must be the same device as the to-be-differentiated tensor's.")
            tensor.grad = upstream_grad.data
        elif isinstance(upstream_grad, np.ndarray) and tensor.is_cpu():
            tensor.grad = upstream_grad
        elif cp is not None and isinstance(upstream_grad, cp.ndarray) and tensor.is_gpu():
            tensor.grad = upstream_grad
        else:
            raise RuntimeError("Upstream gradient must be a numpy/cupy array, or a Tensor.")
    else:
        tensor.grad = tensor.ones_like(tensor.data).data

    for t in topo:
        if t.backward is None:
            if len(t.prev) > 0:
                raise RuntimeError("Tensor is undifferentiable even though it is expected to; backward method is None.")
        else:
            t.backward()
    return topo