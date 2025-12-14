from typing import Literal

from ...typing import (
    TensorType,
    TensorLike,
    DTypeLike,
    Order,
    Axis,
    Device,
    Shape,
)

from ..exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG
from ..utils import to_tensordata, get_device

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


def zeros_like(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
    dtype: DTypeLike | None = None,
    order: Order = "K",
    subok: bool = True,
    shape: Axis | None = None,
) -> TensorType:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device
    a = to_tensordata(a)
    if device_op == "cpu":
        y = np.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape)
    return Tensor(y, requires_grad=requires_grad)


def ones_like(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
    dtype: DTypeLike | None = None,
    order: Order = "K",
    subok: bool = True,
    shape: Axis | None = None,
) -> TensorType:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device
    a = to_tensordata(a)
    if device_op == "cpu":
        y = np.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape)
    return Tensor(y, requires_grad=requires_grad)


def zeros(
    shape: Shape,
    dtype: DTypeLike = float,
    order: Literal["C", "F"] = "C",
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    from ...tensor import Tensor

    if device == "cpu":
        y = np.zeros(shape=shape, dtype=dtype, order=order)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.zeros(shape=shape, dtype=dtype, order=order)
    return Tensor(y, requires_grad=requires_grad)


def ones(
    shape: Shape,
    dtype: DTypeLike = float,
    order: Literal["C", "F"] = "C",
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    from ...tensor import Tensor

    if device == "cpu":
        y = np.ones(shape=shape, dtype=dtype, order=order)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.ones(shape=shape, dtype=dtype, order=order)
    return Tensor(y, requires_grad=requires_grad)


###
###
###

__all__ = ["zeros_like", "ones_like", "zeros", "ones"]
