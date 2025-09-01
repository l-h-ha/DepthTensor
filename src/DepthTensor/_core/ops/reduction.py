from typing import Optional, Union, Any

from ...typing import (
    TensorLike,
    NDArrayLike,
    NDArrayLikeBool,
    Casting,
    Order,
    DTypeLike,
    AxisLike,
    OperandLike,
    DeviceLike,
)

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

from ..utils import to_xp_array, get_device, get_two_operand_op_device

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
    a: OperandLike,
    /,
    *,
    device: Optional[DeviceLike] = None,
    requires_grad: bool = False,
    axis: Optional[AxisLike] = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[Union[np.ndarray, Any]] = None,
    keepdims: bool = True,
    initial: Any = _NoValue,
    where: Union[bool, NDArrayLikeBool] = True,
) -> TensorLike:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    arr = to_xp_array(a, device=device_op)
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
    return Tensor(y, device=device_op, requires_grad=requires_grad)


def max(
    a: OperandLike,
    /,
    *,
    device: Optional[DeviceLike] = None,
    requires_grad: bool = False,
    axis: Optional[AxisLike] = None,
    out: Optional[Union[np.ndarray, Any]] = None,
    keepdims: bool = False,
    initial: Any = _NoValue,
    where: Union[bool, NDArrayLikeBool] = True,
) -> TensorLike:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    arr = to_xp_array(a, device=device_op)
    if device_op == "cpu":
        y = np.max(
            arr, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.max(arr, axis=axis, out=out, keepdims=keepdims)
    return Tensor(y, device=device_op, requires_grad=requires_grad)


def maximum(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[np.ndarray] = None,
    *,
    device: Optional[DeviceLike] = None,
    requires_grad: bool = False,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: Optional[DTypeLike] = None,
    subok: bool = True,
) -> TensorLike:
    from ...tensor import Tensor

    device_op = get_two_operand_op_device(x1, x2, device=device)

    _x1: NDArrayLike = to_xp_array(x1, device=device_op)
    _x2: NDArrayLike = to_xp_array(x2, device=device_op)

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
        y = np.maximum(_x1, _x2, out=out, dtype=dtype, casting=casting)
    return Tensor(y, device=device_op, requires_grad=requires_grad)


###
###
###

__all__ = ["max", "maximum", "sum"]
