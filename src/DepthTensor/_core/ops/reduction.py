from typing import Any

from ...typing import (
    TensorType,
    TensorData,
    TensorDataBool,
    Casting,
    Order,
    DTypeLike,
    Axis,
    TensorLike,
    Device,
)

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

from ..utils import to_tensordata, get_device, get_two_operand_op_device

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None
_NoValue = object()

###
###
###


def sum(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
    axis: Axis | None = None,
    dtype: DTypeLike | None = None,
    out: TensorData | None = None,
    keepdims: bool = True,
    initial: Any = _NoValue,
    where: TensorDataBool | bool = True,
) -> TensorType:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    arr = to_tensordata(a, device=device_op)
    if device_op == "cpu":
        kwds = {"axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        if not isinstance(initial, type(_NoValue)):
            kwds["initial"] = initial
        if out is not None:
            kwds["out"] = out
        y = np.sum(arr, **kwds)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    return Tensor(y, requires_grad=requires_grad)


def max(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
    axis: Axis | None = None,
    out: TensorData | None = None,
    keepdims: bool = False,
    initial: Any = _NoValue,
    where: TensorDataBool | bool = True,
) -> TensorType:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    arr = to_tensordata(a, device=device_op)
    if device_op == "cpu":
        kwargs = {"axis": axis, "out": out, "keepdims": keepdims, "where": where}

        if initial is not _NoValue:
            kwargs["initial"] = initial

        y = np.max(arr, **kwargs)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.max(arr, axis=axis, out=out, keepdims=keepdims)
    return Tensor(y, requires_grad=requires_grad)


def maximum(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    from ...tensor import Tensor

    device_op = get_two_operand_op_device(x1, x2, device=device)

    _x1: TensorData = to_tensordata(x1, device=device_op)
    _x2: TensorData = to_tensordata(x2, device=device_op)

    if device_op == "cpu":
        y = np.maximum(
            _x1,
            _x2,
            out=out,
            dtype=dtype,
            where=where,
            casting=casting,
            order=order,
            subok=subok,
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.maximum(_x1, _x2, out=out, dtype=dtype, casting=casting)
    return Tensor(y, requires_grad=requires_grad)


###
###
###

__all__ = ["max", "maximum", "sum"]
