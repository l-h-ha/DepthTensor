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
    Shape,
)

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

from ..utils import to_tensordata, get_device, get_two_operand_op_device, NoValue

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


def _link_sum_backward(
    y: TensorType, x: TensorLike, axis: Axis | None, keepdims: bool
) -> None:
    from ...tensor import Tensor

    if not isinstance(x, Tensor):
        return

    def callback(y_grad: TensorData, x_shape: Shape, device: Device) -> TensorData:
        if device == "cpu":
            xp = np
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            xp = cp

        grad = y_grad
        if not keepdims and axis is not None:
            grad = xp.expand_dims(grad, axis)

        return xp.broadcast_to(grad, x_shape)

    def backward() -> None:
        if y.grad is None:
            y.zero_grad()
        if x.grad is None:
            x.zero_grad()
        
        # y.grad is TensorData | None, but zero_grad ensures it's not None
        if y.grad is None: return 

        x.grad += callback(y.grad, x.shape, y.device)

    y.prev = (x,)
    y.backward = backward


def sum(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
    axis: Axis | None = None,
    dtype: DTypeLike | None = None,
    out: TensorData | None = None,
    keepdims: bool = True,
    initial: Any = NoValue,
    where: TensorDataBool | bool = True,
) -> TensorType:
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : TensorLike
        Elements to sum.
    device : Device | None, optional
        The device to place the result on. If None, inferred from input.
    requires_grad : bool | None, optional
        Whether the result requires gradient computation. If None, inferred from input.
    axis : Axis | None, optional
        Axis or axes along which a sum is performed.
    dtype : DTypeLike | None, optional
        The type of the returned array and of the accumulator in which
        the elements are summed.
    out : TensorData | None, optional
        Alternative output array in which to place the result.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. Default is True.
    initial : Any, optional
        Starting value for the sum.
    where : TensorDataBool | bool, optional
        Elements to include in the sum.

    Returns
    -------
    TensorType
        An array with the same shape as a, with the specified axis removed.
    """
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    if requires_grad is None:
        requires_grad = a.requires_grad if isinstance(a, Tensor) else False

    arr = to_tensordata(a, device=device_op)
    if device_op == "cpu":
        kwds = {"axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        if initial is not NoValue:
            kwds["initial"] = initial
        if out is not None:
            kwds["out"] = out
        y = np.sum(arr, **kwds)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    
    result = Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)
    
    if requires_grad and isinstance(a, Tensor) and a.requires_grad:
        _link_sum_backward(result, a, axis, keepdims)
        
    return result


def max(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
    axis: Axis | None = None,
    out: TensorData | None = None,
    keepdims: bool = False,
    initial: Any = NoValue,
    where: TensorDataBool | bool = True,
) -> TensorType:
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : TensorLike
        Input data.
    device : Device | None, optional
        The device to place the result on. If None, inferred from input.
    requires_grad : bool | None, optional
        Whether the result requires gradient computation. If None, inferred from input.
    axis : Axis | None, optional
        Axis or axes along which to operate.
    out : TensorData | None, optional
        Alternative output array in which to place the result.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. Default is False.
    initial : Any, optional
        The minimum value of an output element.
    where : TensorDataBool | bool, optional
        Elements to compare.

    Returns
    -------
    TensorType
        Maximum of a.
    """
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    if requires_grad is None:
        requires_grad = a.requires_grad if isinstance(a, Tensor) else False

    arr = to_tensordata(a, device=device_op)
    if device_op == "cpu":
        kwargs = {"axis": axis, "out": out, "keepdims": keepdims, "where": where}

        if initial is not NoValue:
            kwargs["initial"] = initial

        y = np.max(arr, **kwargs)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.max(arr, axis=axis, out=out, keepdims=keepdims)
    return Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)


def maximum(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    """
    Element-wise maximum of array elements.

    Parameters
    ----------
    x1, x2 : TensorLike
        The arrays holding the elements to compare.
    out : TensorData | None, optional
        The results will be placed in this array.
    device : Device | None, optional
        The device to place the result on. If None, inferred from inputs.
    requires_grad : bool | None, optional
        Whether the result requires gradient computation. If None, inferred from inputs.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : DTypeLike | None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        The maximum of x1 and x2, element-wise.
    """
    from ...tensor import Tensor

    x1_is_tensor = isinstance(x1, Tensor)
    x2_is_tensor = isinstance(x2, Tensor)
    device_op = get_two_operand_op_device(
        x1, x2, x1_is_tensor=x1_is_tensor, x2_is_tensor=x2_is_tensor, device=device
    )

    if requires_grad is None:
        requires_grad = False
        if x1_is_tensor:
            requires_grad = requires_grad or x1.requires_grad
        if x2_is_tensor:
            requires_grad = requires_grad or x2.requires_grad

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
    return Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)


###
###
###

__all__ = ["max", "maximum", "sum"]
