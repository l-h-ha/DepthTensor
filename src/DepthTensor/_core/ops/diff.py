from ...typing import (
    TensorLike
)

from ..utils import sum_to_shape
from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG
)

import numpy as np
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
### Arithmetics
###

def add_diff(result: TensorLike, x1: TensorLike, x2: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x1.requires_grad:
            x1.grad += sum_to_shape(result.grad, x1.shape, x1.device)
        if x2.requires_grad:
            x2.grad += sum_to_shape(result.grad, x2.shape, x2.device)
    result.backward = backward
    return result

def subtract_diff(result: TensorLike, x1: TensorLike, x2: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x1.requires_grad:
            x1.grad += sum_to_shape(result.grad, x1.shape, x1.device)
        if x2.requires_grad:
            x2.grad += sum_to_shape(-result.grad, x2.shape, x2.device)
    result.backward = backward
    return result

def multiply_diff(result: TensorLike, x1: TensorLike, x2: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x1.requires_grad:
            x1.grad += sum_to_shape(result.grad * x2.data, x1.shape, x1.device)
        if x2.requires_grad:
            x2.grad += sum_to_shape(result.grad * x1.data, x2.shape, x2.device)
    result.backward = backward
    return result

def matmul_diff(result: TensorLike, x1: TensorLike, x2: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x1.requires_grad:
            x1.grad += sum_to_shape(result.grad @ x2.data.swapaxes(-2,-1), x1.shape, x1.device)
        if x2.requires_grad:
            x2.grad += sum_to_shape(x1.data.swapaxes(-2, -1) @ result.grad, x2.shape, x2.device)
    result.backward = backward
    return result

def divide_diff(result: TensorLike, x1: TensorLike, x2: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x1.requires_grad:
            x1.grad += sum_to_shape(result.grad / x2.data, x1.shape, x1.device)
        if x2.requires_grad:
            x2.grad += sum_to_shape(result.grad * (x1.data * -x2.data**-2), x2.shape, x2.device)
    result.backward = backward
    return result

def power_diff(result: TensorLike, x1: TensorLike, x2: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x1.requires_grad:
            x1.grad += sum_to_shape(result.grad * x2.data*x1.data**(x2.data-1), x1.shape, x1.device)
        if x2.requires_grad:
            if x2.device == "cpu":
                x2.grad += sum_to_shape(result.grad * np.log(x1.data)*x1.data**x2.data, x2.shape, x2.device)
            else:
                if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                x2.grad += sum_to_shape(result.grad * cp.log(x1.data)*x1.data**x2.data, x2.shape, x2.device)
    result.backward = backward
    return result

def negative_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(-result.grad, x.shape, x.device)
    result.backward = backward
    return result

def sign_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(result.grad * 0, x.shape, x.device)
    result.backward = backward
    return result

def abs_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            # TODO: Handle jump discontinuity
            x.grad += sum_to_shape(result.grad * result.data / x.data, x.shape, x.device)
    result.backward = backward
    return result

###
### Exponents/Logarithms
###

def exp_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(result.grad * result.data, x.shape, x.device)
    result.backward = backward
    return result

def sqrt_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(result.grad * (0.5*result.data**(-0.5)), x.shape, x.device)
    result.backward = backward
    return result

def log_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(result.grad / x.data, x.shape, x.device)
    result.backward = backward
    return result

def square_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(result.grad * 2, x.shape, x.device)
    result.backward = backward
    return result

###
###
###

__all__ = [
    'add_diff', 
    'subtract_diff', 
    'multiply_diff', 
    'matmul_diff', 
    'divide_diff',
    'power_diff',
    'negative_diff', 
    'sign_diff', 
    'abs_diff',

    'exp_diff', 
    'sqrt_diff', 
    'log_diff',
    'square_diff'
]