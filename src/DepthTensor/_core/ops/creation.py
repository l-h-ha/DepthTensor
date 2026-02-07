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
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : TensorLike
        The shape and data-type of `a` define these same attributes of the returned array.
    device : Device | None, optional
        The device to place the result on.
    requires_grad : bool, optional
        Whether the result requires gradient computation.
    dtype : DTypeLike | None, optional
        Overrides the data type of the result.
    order : Order, optional
        Overrides the memory layout of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.
    shape : Axis | None, optional
        Overrides the shape of the result.

    Returns
    -------
    TensorType
        Array of zeros with the same shape and type as `a`.
    """
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
    return Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)


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
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : TensorLike
        The shape and data-type of `a` define these same attributes of the returned array.
    device : Device | None, optional
        The device to place the result on.
    requires_grad : bool, optional
        Whether the result requires gradient computation.
    dtype : DTypeLike | None, optional
        Overrides the data type of the result.
    order : Order, optional
        Overrides the memory layout of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.
    shape : Axis | None, optional
        Overrides the shape of the result.

    Returns
    -------
    TensorType
        Array of ones with the same shape and type as `a`.
    """
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
    return Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)


def zeros(
    shape: Shape,
    dtype: DTypeLike = float,
    order: Literal["C", "F"] = "C",
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : Shape
        Shape of the new array.
    dtype : DTypeLike, optional
        The desired data-type for the array. Default is float.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major (C-style)
        or column-major (Fortran-style) order in memory.
    device : Device, optional
        The device to place the result on. Default is 'cpu'.
    requires_grad : bool, optional
        Whether the result requires gradient computation.

    Returns
    -------
    TensorType
        Array of zeros with the given shape, dtype, and order.
    """
    from ...tensor import Tensor

    if device == "cpu":
        y = np.zeros(shape=shape, dtype=dtype, order=order)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.zeros(shape=shape, dtype=dtype, order=order)
    return Tensor._fast_init(y, device=device, requires_grad=requires_grad)


def ones(
    shape: Shape,
    dtype: DTypeLike = float,
    order: Literal["C", "F"] = "C",
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : Shape
        Shape of the new array.
    dtype : DTypeLike, optional
        The desired data-type for the array. Default is float.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major (C-style)
        or column-major (Fortran-style) order in memory.
    device : Device, optional
        The device to place the result on. Default is 'cpu'.
    requires_grad : bool, optional
        Whether the result requires gradient computation.

    Returns
    -------
    TensorType
        Array of ones with the given shape, dtype, and order.
    """
    from ...tensor import Tensor

    if device == "cpu":
        y = np.ones(shape=shape, dtype=dtype, order=order)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.ones(shape=shape, dtype=dtype, order=order)
    return Tensor._fast_init(y, device=device, requires_grad=requires_grad)


###
###
###

__all__ = ["zeros_like", "ones_like", "zeros", "ones"]
