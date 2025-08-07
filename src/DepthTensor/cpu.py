from __future__ import annotations
from typing import Any, Optional, Tuple, Union

from ._core.api import api
from ._core import sum_to_shape
from .types import DTypeLike

import numpy as np

class cpu(api):
    def __init__(self, data: Any, dtype: Optional[DTypeLike] = None, prev: Tuple = (), requires_grad: bool = False) -> None:
        super().__init__(data, "cpu", dtype, prev, requires_grad)
        self.grad = np.zeros_like(self.data, self.dtype)

    @property
    def T(self) -> cpu:
        return cpu(data=self.data.T, dtype=self.dtype)

    ###
    ###
    ###

    @classmethod
    def zeros_like(cls, a: Union[cpu, Any], dtype: Optional[DTypeLike] = None) -> Any:
        if isinstance(a, cpu):
            return cls(data=np.zeros_like(a.data, dtype=dtype), dtype=dtype)
        return cls(data=np.zeros_like(a, dtype=dtype), dtype=dtype)
    
    @classmethod
    def ones_like(cls, a: Union[cpu, Any], dtype: Optional[DTypeLike] = None) -> Any:
        if isinstance(a, cpu):
            return cls(data=np.ones_like(a.data, dtype=dtype), dtype=dtype)
        return cls(data=np.ones_like(a, dtype=dtype), dtype=dtype)


    ###
    ###
    ###

    def add(self, x1: cpu, x2: cpu, produce_tensor: bool = True) -> Union[cpu, np.ndarray]:
        y = np.add(x1.data, x2.data)
        if produce_tensor:
            return self.add_diff(cpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def add_diff(self, result: cpu, x1: cpu, x2: cpu) -> cpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(result.grad, x2.shape, result.device)
        result.backward = backward
        return result
    
    def subtract(self, x1: cpu, x2: cpu, produce_tensor: bool = True) -> Union[cpu, np.ndarray]:
        y = np.subtract(x1.data, x2.data)
        if produce_tensor:
            return self.subtract_diff(cpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def subtract_diff(self, result: cpu, x1: cpu, x2: cpu) -> cpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(-result.grad, x2.shape, result.device)
        result.backward = backward
        return result
    
    def multiply(self, x1: cpu, x2: cpu, produce_tensor: bool = True) -> Union[cpu, np.ndarray]:
        y = np.multiply(x1.data, x2.data)
        if produce_tensor:
            return self.multiply_diff(cpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def multiply_diff(self, result: cpu, x1: cpu, x2: cpu) -> cpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad * x2.data, x1.shape, result.device)
            if x2.requires_grad:
                x2.grad += sum_to_shape(result.grad * x1.data, x2.shape, result.device)
        result.backward = backward
        return result
    
    def matmul(self, x1: cpu, x2: cpu, produce_tensor: bool = True) -> Union[cpu, np.ndarray]:
        y = np.matmul(x1.data, x2.data)
        if produce_tensor:
            return self.matmul_diff(cpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def matmul_diff(self, result: cpu, x1: cpu, x2: cpu) -> cpu:
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
    
    def divide(self, x1: cpu, x2: cpu, produce_tensor: bool = True) -> Union[cpu, np.ndarray]:
        y = np.divide(x1.data, x2.data)
        if produce_tensor:
            return self.divide_diff(cpu(data=y, prev=(x1, x2), requires_grad=x1.requires_grad or x2.requires_grad), x1, x2)
        return y
    def divide_diff(self, result: cpu, x1: cpu, x2: cpu) -> cpu:
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
            return self.negative_diff(cpu(data=-self.data, dtype=dtype, prev=(self,), requires_grad=self.requires_grad), self)
        return y
    def negative_diff(self, result: cpu, x1: cpu) -> cpu:
        if not result.requires_grad: return result
        def backward() -> None:
            if x1.requires_grad:
                x1.grad += sum_to_shape(result.grad * -1, x1.shape, result.device)
        result.backward = backward
        return result