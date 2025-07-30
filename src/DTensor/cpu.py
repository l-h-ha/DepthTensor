from __future__ import annotations
from typing import Any, Optional, Tuple

from ._core.api import api

import numpy as np

class cpu(api):
    def __init__(self, data: Any, dtype: Optional[np.typing.DTypeLike] = None, prev: Tuple = (), requires_grad: bool = False) -> None:
        super().__init__(data, "cpu", dtype, prev, requires_grad)
        self.grad = np.zeros_like(self.data, self.dtype)

    ###
    ###
    ###

    def add(self, x1: cpu, x2: cpu) -> api:
        return cpu(data=np.add(x1.data, x2.data))
    
    def subtract(self, x1: cpu, x2: cpu) -> api:
        return cpu(np.subtract(x1.data, x2.data))
    
    def multiply(self, x1: cpu, x2: cpu) -> api:
        return cpu(np.multiply(x1.data, x2.data))
    
    def matmul(self, x1: cpu, x2: cpu) -> api:
        return cpu(np.matmul(x1.data, x2.data))
    
    def divide(self, x1: cpu, x2: cpu) -> api:
        return cpu(np.divide(x1.data, x2.data))