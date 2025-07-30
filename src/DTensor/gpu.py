from __future__ import annotations
from typing import Any, Optional, Tuple

from ._core.api import api

import numpy as np
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None
CUPY_NOT_FOUND = "Module CuPy not found or installed."

class gpu(api):
    def __init__(self, data: Any, dtype: Optional[np.typing.DTypeLike] = None, prev: Tuple = (), requires_grad: bool = False) -> None:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        super().__init__(data, "gpu", dtype, prev, requires_grad)
        self.grad = cp.zeros_like(self.data, self.dtype)

    ###
    ###
    ###

    def add(self, x1: gpu, x2: gpu) -> gpu:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        return gpu(cp.add(x1.data, x2.data))
    
    def subtract(self, x1: gpu, x2: gpu) -> gpu:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        return gpu(cp.subtract(x1.data, x2.data))
    
    def multiply(self, x1: gpu, x2: gpu) -> gpu:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        return gpu(cp.multiply(x1.data, x2.data))
    
    def matmul(self, x1: gpu, x2: gpu) -> gpu:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        return gpu(cp.matmul(x1.data, x2.data))
    
    def divide(self, x1: gpu, x2: gpu) -> gpu:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        return gpu(cp.divide(x1.data, x2.data))