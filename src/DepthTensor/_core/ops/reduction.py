from typing import (
    Optional, 
    Union, 
    Any
)

from ...typing import (
    TensorLike, 
    ShapeLike,
    ArrayLikeBool,
    Casting,
    Order,
    DTypeLike,
    AxisShapeLike,
    OperandLike,
    DeviceLike
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG,
    DeviceMismatch, DEVICE_MISMATCH_MSG
)

from ..utils import (
    to_xp_array
)

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
    device: DeviceLike = "cpu",
    axis: Optional[AxisShapeLike] = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[Union[np.ndarray, Any]] = None,
    keepdims: bool = True,
    initial: Any = _NoValue,
    where: Union[bool, ArrayLikeBool] = True
) -> TensorLike:
    from ...tensor import Tensor
    if isinstance(a, Tensor):
        device_op = a.device
    else:
        device_op = device

    arr = to_xp_array(a, device=device_op)
    if device_op == "cpu":
        kwds = {
            'axis': axis,
            'dtype': dtype,
            'keepdims': keepdims,
            'where': where
        }
        if not isinstance(initial, type(_NoValue)):
            kwds['initial'] = initial        
        if out is not None:
            kwds['out'] = out
        y = np.sum(arr, **kwds)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    return Tensor(y, device=device_op)

def max(
    a: OperandLike,
    /,
    *,
    device: DeviceLike = "cpu",
    axis: Optional[ShapeLike] = None, 
    out: Optional[Union[np.ndarray, Any]] = None, 
    keepdims: bool = False, 
    initial: Any = _NoValue, 
    where: Union[bool, ArrayLikeBool] = True
) -> TensorLike:
    from ...tensor import Tensor
    if isinstance(a, Tensor):
        device_op = a.device
    else:
        device_op = device

    arr = to_xp_array(a, device=device_op)
    if device_op == "cpu":
        y = np.max(arr, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.max(arr, axis=axis, out=out, keepdims=keepdims)
    return Tensor(y, device=device_op)

def maximum(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[np.ndarray] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: Optional[DTypeLike] = None,
    subok: bool = True
) -> TensorLike:
    from ...tensor import Tensor
    is_tensor_op = False
    if isinstance(x1, Tensor) and isinstance(x2, Tensor):
        if not x1.is_device(x2.device): raise DeviceMismatch(DEVICE_MISMATCH_MSG)
        is_tensor_op = True

    if is_tensor_op and isinstance(x1, Tensor):
        device_op = x1.device
    else:
        device_op = device

    x1, x2 = to_xp_array(x1, device=device_op), to_xp_array(x2, device=device_op)
    if device_op == "cpu":
        y = np.maximum(x1, x2, out=out, dtype=dtype, where=where, casting=casting, order=order, subok=subok)
    else:
        y = np.maximum(x1, x2, out=out, dtype=dtype, casting=casting)
    return Tensor(y, device=device_op)

###
###
###

__all__ = [
    'max', 
    'maximum', 
    'sum'
]