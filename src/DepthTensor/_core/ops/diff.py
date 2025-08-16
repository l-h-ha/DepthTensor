from ...typing import (
    TensorLike
)

from ..utils import sum_to_shape

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

def negative_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    if not result.requires_grad: return result
    def backward() -> None:
        if x.requires_grad:
            x.grad += sum_to_shape(-result.grad, x.shape, x.device)
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

###
###
###

__all__ = [
    'add_diff', 'subtract_diff', 'multiply_diff', 'matmul_diff', 'divide_diff',
    'negative_diff', 'exp_diff'
]