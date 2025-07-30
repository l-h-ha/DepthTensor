from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

class api(ABC):
    """
    Abstract-base-class numpy / cupy wrapper.

    Defines behaviors shared by the two libraries.
    """

    def __init__(self, data: Any, device: str, dtype: Optional[np.typing.DTypeLike] = None, prev: Tuple = (), requires_grad: bool = False) -> None:
        super().__init__()
        if isinstance(data, (int, float, list, tuple)):
            data = np.asarray(data)
        else:
            raise RuntimeError(f"Expected an object of type: int, float, list, tuple, np.ndarray, np.floating, np.integer. Got: {type(data)}")
        if dtype is not None and data.dtype != dtype:
            data = data.astype(dtype=dtype)
        self.data = data
        self.device = device
        self.dtype = data.dtype
        self.prev = prev
        self.requires_grad = requires_grad
        self.backward  = None

    def get_device(self) -> str:
        return self.device
    
    ###
    ###
    ###

    @abstractmethod
    def add(self, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def subtract(self, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def multiply(self, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def matmul(self, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def divide(self, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    ###
    ###
    ###

    def __add__(self: Any, x2: Any) -> Any:
        return self.add(self.data, x2)
    def __radd__(self: Any, x2: Any) -> Any:
        return self.add(x2, self.data)
    def __iadd__(self: Any, x2: Any) -> Any:
        self.data = self.add(self.data, x2)
        return self
    
    def __sub__(self: Any, x2: Any) -> Any:
        return self.subtract(self.data, x2)
    def __rsub__(self: Any, x2: Any) -> Any:
        return self.subtract(x2, self.data)
    def __isub__(self: Any, x2: Any) -> Any:
        self.data = self.subtract(self.data, x2)
        return self
    
    def __mul__(self: Any, x2: Any) -> Any:
        return self.multiply(self.data, x2)
    def __rmul__(self: Any, x2: Any) -> Any:
        return self.multiply(x2, self.data)
    def __imul__(self: Any, x2: Any) -> Any:
        self.data = self.multiply(self.data, x2)
        return self
    
    def __truediv__(self: Any, x2: Any) -> Any:
        return self.divide(self.data, x2)
    def __rtruediv__(self: Any, x2: Any) -> Any:
        return self.divide(x2, self.data)
    def __itruediv__(self: Any, x2: Any) -> Any:
        self.data = self.divide(self.data, x2)
        return self
    
    def __matmul__(self: Any, x2: Any) -> Any:
        return self.matmul(self.data, x2)
    def __rmatmul__(self: Any, x2: Any) -> Any:
        return self.matmul(x2, self.data)
    def __imatmul__(self: Any, x2: Any) -> Any:
        self.data = self.matmul(self.data, x2)
        return self