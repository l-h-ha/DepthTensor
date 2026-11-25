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

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
)
from ..utils import _check_device, _check_dtype, to_tensordata, get_device

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

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
    Create zeros from a template.

    Parameters
    ----------
    a : TensorLike
        Template for shape, dtype, device, and layout.
    device : {'cpu', 'cuda'}, optional
        If None, use the input device.
    requires_grad : bool, optional
        Track gradients for a floating-point tensor.
    dtype : DTypeLike, optional
        Override the template dtype.
    order : Order, optional
        Output layout. 'K' matches the input layout where possible.
    subok : bool, optional
        Preserve NumPy array subclasses.
    shape : int or tuple of int, optional
        Override the template dimensions.

    Returns
    -------
    Tensor
        Tensor filled with zeros.

    Raises
    ------
    RuntimeError
        Gradient tracking requested for an integer or Boolean result.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor

    _check_dtype(dtype)

    if device is None:
        op_device = get_device(a)
    else:
        op_device = device
    a = to_tensordata(a, device=op_device)

    if op_device == "cpu":
        y = np.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape)

    return Tensor._fast_init(y, device=op_device, requires_grad=requires_grad)

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
    Create ones from a template.

    Parameters
    ----------
    a : TensorLike
        Template for shape, dtype, device, and layout.
    device : {'cpu', 'cuda'}, optional
        If None, use the input device.
    requires_grad : bool, optional
        Track gradients for a floating-point tensor.
    dtype : DTypeLike, optional
        Override the template dtype.
    order : Order, optional
        Output layout. 'K' matches the input layout where possible.
    subok : bool, optional
        Preserve NumPy array subclasses.
    shape : int or tuple of int, optional
        Override the template dimensions.

    Returns
    -------
    Tensor
        Tensor filled with ones.

    Raises
    ------
    RuntimeError
        Gradient tracking requested for an integer or Boolean result.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor

    _check_dtype(dtype)

    if device is None:
        op_device = get_device(a)
    else:
        op_device = device
    a = to_tensordata(a, device=op_device)

    if op_device == "cpu":
        y = np.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = cp.ones_like(a, dtype=dtype, order=order, subok=None, shape=shape)

    return Tensor._fast_init(y, device=op_device, requires_grad=requires_grad)

def zeros(
    shape: Shape,
    dtype: DTypeLike = float,
    order: Literal["C", "F"] = "C",
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    """
    Create a tensor of zeros.

    Parameters
    ----------
    shape : tuple of int
        Output dimensions.
    dtype : DTypeLike, optional
        Output dtype. Defaults to float.
    order : {'C', 'F'}, optional
        Row-major or column-major layout.
    device : {'cpu', 'cuda'}, optional
        Defaults to CPU.
    requires_grad : bool, optional
        Track gradients for a floating-point tensor.

    Returns
    -------
    Tensor
        Tensor filled with zeros.

    Raises
    ------
    RuntimeError
        Gradient tracking requested for an integer or Boolean result.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor

    _check_dtype(dtype)
    _check_device(device)

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
    Create a tensor of ones.

    Parameters
    ----------
    shape : tuple of int
        Output dimensions.
    dtype : DTypeLike, optional
        Output dtype. Defaults to float.
    order : {'C', 'F'}, optional
        Row-major or column-major layout.
    device : {'cpu', 'cuda'}, optional
        Defaults to CPU.
    requires_grad : bool, optional
        Track gradients for a floating-point tensor.

    Returns
    -------
    Tensor
        Tensor filled with ones.

    Raises
    ------
    RuntimeError
        Gradient tracking requested for an integer or Boolean result.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor

    _check_dtype(dtype)
    _check_device(device)

    if device == "cpu":
        y = np.ones(shape=shape, dtype=dtype, order=order)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = cp.ones(shape=shape, dtype=dtype, order=order)

    return Tensor._fast_init(y, device=device, requires_grad=requires_grad)
