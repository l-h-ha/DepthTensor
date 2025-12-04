from typing import Callable

from ...typing import OperandLike, TensorLike, NDArrayLike, AxisLike

from ..utils import sum_to_shape, to_xp_array, get_two_operand_op_device
from ..exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG

import numpy as np
import math

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
        if result.grad is None:
            result.zeros_grad()

        result_grad: NDArrayLike = result.grad
        device = get_two_operand_op_device(x1, x2, None)
        x1_data, x2_data = to_xp_array(x1, device=device), to_xp_array(
            x2, device=device
        )

        if isinstance(x1, Tensor) and x1.requires_grad:
            if x1.grad is None:
                x1.zeros_grad()
            x1.grad += callback_x1(
                result_grad, x1.shape, device, x1_data, x2_data
            ).astype(x1.dtype)
        if isinstance(x2, Tensor) and x2.requires_grad:
            if x2.grad is None:
                x2.zeros_grad()
            x2.grad += callback_x2(
                result_grad, x2.shape, device, x1_data, x2_data
            ).astype(x2.dtype)

    result.backward = backward
    return result


def wrapper_1in_diff(
    result: TensorLike, x: OperandLike, callback_x1: Callable
) -> TensorLike:
    if not result.requires_grad:
        return result
    from ...tensor import Tensor

    def backward() -> None:
        if result.grad is None:
            result.zeros_grad()

        result_grad: NDArrayLike = result.grad
        _x = to_xp_array(x)
        if isinstance(x, Tensor) and x.requires_grad:
            if x.grad is None:
                x.zeros_grad()
            x.grad += callback_x1(result_grad, x.shape, x.device, _x)

    result.backward = backward
    return result


###
### Arithmetics
###


def add_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:
    def callback_x1(result_grad, x1_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad, x1_shape, device)

    def callback_x2(result_grad, x2_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad, x2_shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def subtract_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad, x1_shape, device)

    def callback_x2(result_grad, x2_shape, device, x1_data, x2_data):
        return sum_to_shape(-result_grad, x2_shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def multiply_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad * x2_data, x1_shape, device)

    def callback_x2(result_grad, x2_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad * x1_data, x2_shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def matmul_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad @ x2_data.swapaxes(-2, -1), x1_shape, device)

    def callback_x2(result_grad, x2_shape, device, x1_data, x2_data):
        return sum_to_shape(x1_data.swapaxes(-2, -1) @ result_grad, x2_shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def divide_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad / x2_data, x1_shape, device)

    def callback_x2(result_grad, x2_shape, device, x1_data, x2_data):
        return sum_to_shape(result_grad * (x1_data * -(x2_data**-2)), x2_shape, device)

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def power_diff(result: TensorLike, x1: OperandLike, x2: OperandLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data, x2_data):
        return sum_to_shape(
            result_grad * x2_data * x1_data ** (x2_data - 1), x1_shape, device
        )

    def callback_x2(result_grad, x2_shape, device, x1_data, x2_data):
        if device == "cpu":
            return sum_to_shape(
                result_grad * np.log(x1_data) * x1_data**x2_data, x2_shape, device
            )
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            return sum_to_shape(
                result_grad * cp.log(x1_data) * x1_data**x2_data, x2_shape, device
            )

    return wrapper_2in_diff(result, x1, x2, callback_x1, callback_x2)


def negative_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(-result_grad, x1_shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def sign_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(result_grad * 0, x1_shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def abs_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(result_grad * result.data / x1_data, x1_shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


###
### Exponents/Logarithms
###


def exp_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(result_grad * result.data, x1_shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def sqrt_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(
            result_grad * (0.5 * result.data ** (-0.5)), x1_shape, device
        )

    return wrapper_1in_diff(result, x, callback_x1)


def log_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(result_grad / x1_data, x1_shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


def square_diff(result: TensorLike, x: TensorLike) -> TensorLike:

    def callback_x1(result_grad, x1_shape, device, x1_data):
        return sum_to_shape(result_grad * 2 * x1_data, x1_shape, device)

    return wrapper_1in_diff(result, x, callback_x1)


###
###
###


def mean_diff(
    result: TensorLike, x: TensorLike, axis: AxisLike, keepdims: bool
) -> TensorLike:
    def callback_x1(result_grad, x1_shape, device, _x):
        if device == "cpu":
            xp = np
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            xp = cp

        inp_size = math.prod(x1_shape)
        grad_size = result_grad.size
        N = inp_size / grad_size
        grad = result_grad * (1.0 / N)

        if not keepdims and axis is not None:
            grad = xp.expand_dims(grad, axis)
        return xp.broadcast_to(grad, x1_shape)

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
    "mean_diff",
]
