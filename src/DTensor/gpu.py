from __future__ import annotations
from typing import Any, Optional, Tuple, Union

from ._core.api import api
from ._core import sum_to_shape
from .types import DTypeLike

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None
CUPY_NOT_FOUND = "Module CuPy not found or installed."

class gpu(api):
    def __init__(self, data: Any, dtype: Optional[DTypeLike] = None, prev: Tuple = (), requires_grad: bool = False) -> None:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        super().__init__(data, "gpu", dtype, prev, requires_grad)
        self.grad = cp.zeros_like(self.data, self.dtype)

    @property
    def T(self) -> gpu:
        return gpu(data=self.data.T, dtype=self.dtype)

    ###
    ###
    ###

    @classmethod
    def zeros_like(cls, a: Union[gpu, Any], dtype: Optional[DTypeLike] = None) -> Any:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        if isinstance(a, gpu):
            return cls(data=cp.zeros_like(a.data, dtype=dtype), dtype=dtype)
        return cls(data=cp.zeros_like(a, dtype=dtype), dtype=dtype)
    
    @classmethod
    def ones_like(cls, a: Union[gpu, Any], dtype: Optional[DTypeLike] = None) -> Any:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        if isinstance(a, gpu):
            return cls(data=cp.ones_like(a.data, dtype=dtype), dtype=dtype)
        return cls(data=cp.ones_like(a, dtype=dtype), dtype=dtype)

    ###
    ###
    ###

    def add(self, x1: gpu, x2: gpu, produce_tensor: bool = False) -> Union[gpu, Any]:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        y = cp.add(x1.data, x2.data)
        if produce_tensor:
            return self.add_diff(gpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def add_diff(self, result: gpu, x1: gpu, x2: gpu) -> gpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad * 1, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(result.grad * 1, x2.shape, result.device)
        result.backward = backward
        return result
    
    def subtract(self, x1: gpu, x2: gpu, produce_tensor: bool = False) -> Union[gpu, Any]:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        y = cp.subtract(x1.data, x2.data)
        if produce_tensor:
            return self.subtract_diff(gpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def subtract_diff(self, result: gpu, x1: gpu, x2: gpu) -> gpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad * 1, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(result.grad * -1, x2.shape, result.device)
        result.backward = backward
        return result
    
    def multiply(self, x1: gpu, x2: gpu, produce_tensor: bool = False) -> Union[gpu, Any]:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        y = cp.multiply(x1.data, x2.data)
        if produce_tensor:
            return self.multiply_diff(gpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def multiply_diff(self, result: gpu, x1: gpu, x2: gpu) -> gpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad * x2.data, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(result.grad * x1.data, x2.shape, result.device)
        result.backward = backward
        return result
    
    def matmul(self, x1: gpu, x2: gpu, produce_tensor: bool = False) -> Union[gpu, Any]:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        y = cp.matmul(x1.data, x2.data)
        if produce_tensor:
            return self.matmul_diff(gpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def matmul_diff(self, result: gpu, x1: gpu, x2: gpu) -> gpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.ndim > 1 and x2.ndim > 1:
                if x1.requires_grad:
                    x1.grad += sum_to_shape(result.grad @ x2.data.swapaxes(-2, -1), x1.shape, result.device)
                if x2.requires_grad:
                    x2.grad += sum_to_shape(x1.data.swapaxes(-2, -1) @ result.grad, x2.shape, result.device)
            else:
                raise RuntimeError("Performing matrix multiplication requires objects of dimensions higher than a vector.")
        result.backward = backward
        return result
    
    def divide(self, x1: gpu, x2: gpu, produce_tensor: bool = False) -> Union[gpu, Any]:
        if cp is None: raise RuntimeError(CUPY_NOT_FOUND)
        y = cp.divide(x1.data, x2.data)
        if produce_tensor:
            return self.divide_diff(gpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def divide_diff(self, result: gpu, x1: gpu, x2: gpu) -> gpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad / x2.data, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(result.grad * (x1.data * -x2.data**(-2)), x2.shape, result.device)
        result.backward = backward
        return result
    
    ###
    ###
    ###

    def negative(self, dtype: Optional[DTypeLike] = None, produce_tensor: bool = False) -> Any:
        y = -self.data
        if produce_tensor:
            return self.negative_diff(gpu(data=-self.data, dtype=dtype, prev=(self,), requires_grad=self.requires_grad), self)
        return y
    def negative_diff(self, result: gpu, x1: gpu) -> gpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad * -1, x1.shape, result.device)
        result.backward = backward
        return result