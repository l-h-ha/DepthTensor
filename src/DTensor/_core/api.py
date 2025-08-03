from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Callable, TypeAlias

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None
DTypeLike: TypeAlias = np.typing.DTypeLike

class api(ABC):
    """
    Abstract-base-class numpy / cupy wrapper.

    Defines behaviors shared by the two libraries.
    """

    def __init__(self, data: Any, device: str, dtype: Optional[DTypeLike] = None, prev: Tuple = (), requires_grad: bool = False) -> None:
        super().__init__()
        if isinstance(data, (int, float, list, tuple, np.ndarray, np.floating, np.integer)):
            data = np.asarray(data)
        elif cp is not None and isinstance(data, (cp.ndarray, cp.floating, cp.integer)):
            data = cp.asarray(data)
        else:
            raise RuntimeError(f"Expected an object of type: int, float, list, tuple, numpy.ndarray, numpy.floating, numpy.integer, cupy.ndarray, cupy.floating, cupy.integer. Got: {type(data)}")
        if dtype is not None and data.dtype != dtype:
            data = data.astype(dtype=dtype)
        self.data = data
        self.device = device
        self.dtype = data.dtype
        self.prev = prev
        self.requires_grad = requires_grad
        self.backward: Optional[Callable[[], None]] = None

    def get_device(self) -> str:
        return self.device
    def is_device(self, device: str) -> bool:
        return self.device == device
    def is_cpu(self) -> bool:
        return self.device == "cpu"
    def is_gpu(self) -> bool:
        return self.device == "gpu"
    
    @property
    def shape(self) -> Tuple:
        return self.data.shape
    @property
    def ndim(self) -> int:
        return self.data.ndim
    @property
    def size(self) -> int:
        return self.data.size
    @property
    def T(self) -> Any:
        raise NotImplementedError
    
    ###
    ###
    ###

    @classmethod
    @abstractmethod
    def zeros_like(cls, a: Any, dtype: Optional[DTypeLike] = None) -> Any:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def ones_like(cls, a: Any, dtype: Optional[DTypeLike] = None) -> Any:
        raise NotImplementedError

    ###
    ###
    ###

    @abstractmethod
    def add(self, x1: Any, x2: Any, produce_tensor: bool = True) -> Any:
        raise NotImplementedError
    @abstractmethod
    def add_diff(self, result: Any, x1: Any, x2: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def subtract(self, x1: Any, x2: Any, produce_tensor: bool = True) -> Any:
        raise NotImplementedError
    @abstractmethod
    def subtract_diff(self, result: Any, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def multiply(self, x1: Any, x2: Any, produce_tensor: bool = True) -> Any:
        raise NotImplementedError
    @abstractmethod
    def multiply_diff(self, result: Any, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def matmul(self, x1: Any, x2: Any, produce_tensor: bool = True) -> Any:
        raise NotImplementedError
    @abstractmethod
    def matmul_diff(self, result: Any, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def divide(self, x1: Any, x2: Any, produce_tensor: bool = True) -> Any:
        raise NotImplementedError
    @abstractmethod
    def divide_diff(self, result: Any, x1: Any, x2: Any) -> Any:
        raise NotImplementedError
    
    ###
    ###
    ###

    @abstractmethod
    def negative(self, dtype: Optional[DTypeLike] = None, produce_tensor: bool = False) -> Any:
        raise NotImplementedError

    ###
    ###
    ###

    def __add__(self: Any, x2: Any) -> Any:
        return self.add(self, x2)
    def __radd__(self: Any, x2: Any) -> Any:
        return self.add(x2, self)
    def __iadd__(self: Any, x2: Any) -> Any:
        self.data = self.add(self, x2, produce_tensor = False)
        return self
    
    def __sub__(self: Any, x2: Any) -> Any:
        return self.subtract(self, x2)
    def __rsub__(self: Any, x2: Any) -> Any:
        return self.subtract(x2, self)
    def __isub__(self: Any, x2: Any) -> Any:
        self.data = self.subtract(self, x2, produce_tensor = False)
        return self
    
    def __mul__(self: Any, x2: Any) -> Any:
        return self.multiply(self.data, x2)
    def __rmul__(self: Any, x2: Any) -> Any:
        return self.multiply(x2, self.data)
    def __imul__(self: Any, x2: Any) -> Any:
        self.data = self.multiply(self.data, x2, produce_tensor = False)
        return self
    
    def __truediv__(self: Any, x2: Any) -> Any:
        return self.divide(self.data, x2)
    def __rtruediv__(self: Any, x2: Any) -> Any:
        return self.divide(x2, self.data)
    def __itruediv__(self: Any, x2: Any) -> Any:
        self.data = self.divide(self.data, x2, produce_tensor = False)
        return self
    
    def __matmul__(self: Any, x2: Any) -> Any:
        return self.matmul(self.data, x2)
    def __rmatmul__(self: Any, x2: Any) -> Any:
        return self.matmul(x2, self.data)
    def __imatmul__(self: Any, x2: Any) -> Any:
        self.data = self.matmul(self.data, x2, produce_tensor = False)
        return self
    
    ###
    ###
    ###

    def __neg__(self) -> Any:
        return self.negative()

    ###
    ###
    ###

    def __repr__(self) -> str:
        return f"{self.device.upper()}tensor({self.data})"
    
    def __getitem__(self, index) -> Any:
        return self.data[index]
    
    def __setitem__(self, index, value) -> None:
        self.data[index] = value