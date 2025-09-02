from typing import Callable

from ...typing import OperandLike, TensorLike

from ..utils import sum_to_shape, to_xp_array, get_two_operand_op_device
from ..exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


def wrapper_2in_diff(
    result: TensorLike,
    x1: OperandLike,
    x2: OperandLike,
    callback_x1: Callable,
    callback_x2: Callable,
) -> TensorLike:
    if not result.requires_grad:
        return result
    from ...tensor import Tensor

    def backward() -> None:
        device = get_two_operand_op_device(x1, x2, None)
        _x1, _x2 = to_xp_array(x1, device=device), to_xp_array(x2, device=device)

        if isinstance(x1, Tensor) and x1.requires_grad:
            x1.grad += callback_x1(x1.shape, device, _x1, _x2)
        if isinstance(x2, Tensor) and x2.requires_grad:
            x2.grad += callback_x2(x2.shape, device, _x1, _x2)

    result.backward = backward
    return result


def wrapper_1in_diff(
    result: TensorLike, x: OperandLike, callback_x1: Callable
) -> TensorLike:
    if not result.requires_grad:
        return result
    from ...tensor import Tensor

    def backward() -> None:
        _x = to_xp_array(x)
        if isinstance(x, Tensor) and x.requires_grad:
            x.grad += callback_x1(x.shape, x.device, _x)

    result.backward = backward
    return result


###
### Arithmetics
###


def add_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(shape, device, _x1, _x2):
        return sum_to_shape(result.grad, shape, device)

    def callback_x2(shape, device, _x1, _x2):
        return sum_to_shape(result.grad, shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def subtract_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(shape, device, _x1, _x2):
        return sum_to_shape(result.grad, shape, device)

    def callback_x2(shape, device, _x1, _x2):
        return sum_to_shape(-result.grad, shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def multiply_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(shape, device, _x1, _x2):
        return sum_to_shape(result.grad * _x2, shape, device)

    def callback_x2(shape, device, _x1, _x2):
        return sum_to_shape(result.grad * _x1, shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def matmul_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(shape, device, _x1, _x2):
        return sum_to_shape(result.grad @ _x2.swapaxes(-2, -1), shape, device)

    def callback_x2(shape, device, _x1, _x2):
        return sum_to_shape(_x1.swapaxes(-2, -1) @ result.grad, shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def divide_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(shape, device, _x1, _x2):
        return sum_to_shape(result.grad / _x2, shape, device)

    def callback_x2(shape, device, _x1, _x2):
        return sum_to_shape(result.grad * (_x1 * -(_x2**-2)), shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def power_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(shape, device, _x1, _x2):
        return sum_to_shape(result.grad * _x2 * _x1 ** (_x2 - 1), shape, device)

    def callback_x2(shape, device, _x1, _x2):
        if device == "cpu":
            return sum_to_shape(result.grad * np.log(_x1) * _x1**_x2, shape, device)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            return sum_to_shape(result.grad * cp.log(_x1) * _x1**_x2, shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def negative_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(-result.data, shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def sign_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(result.grad * 0, shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def abs_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(result.grad * result.data / _x, shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


###
### Exponents/Logarithms
###


def exp_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(result.grad * result.data, shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def sqrt_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(result.grad * (0.5 * result.data ** (-0.5)), shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def log_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(result.grad / _x, shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def square_diff(result: TensorLike, x: TensorLike) -> TensorLike:
    def callback_x1(shape, device, _x):
        return sum_to_shape(result.grad * 2 * _x, shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


###
###
###

__all__ = [
    "add_diff",
    "subtract_diff",
    "multiply_diff",
    "matmul_diff",
    "divide_diff",
    "power_diff",
    "negative_diff",
    "sign_diff",
    "abs_diff",
    "exp_diff",
    "sqrt_diff",
    "log_diff",
    "square_diff",
]
