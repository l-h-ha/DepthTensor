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
    DTypeLike
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG,
    DeviceMismatch, DEVICE_MISMATCH_MSG
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

def max(
    a: TensorLike,
    /,
    *,
    axis: Optional[ShapeLike] = None, 
    out: Optional[Union[np.ndarray, Any]] = None, 
    keepdims: bool = False, 
    initial: Any = _NoValue, 
    where: Union[bool, ArrayLikeBool] = True
) -> TensorLike:
    from ...tensor import Tensor
    if a.is_cpu():
        y = np.max(a.data, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.max(a.data, axis=axis, out=out, keepdims=keepdims)
    return Tensor(y, device=a.device)

def maximum(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[np.ndarray] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: Optional[DTypeLike] = None,
    subok: bool = True
) -> TensorLike:
    from ...tensor import Tensor
    if not x1.is_device(x2.device): raise DeviceMismatch(DEVICE_MISMATCH_MSG)
    if x1.is_cpu():
        y = np.maximum(x1.data, x2.data, out=out, dtype=dtype, where=where, casting=casting, order=order, subok=subok)
    else:
        y = np.maximum(x1.data, x2.data, out=out, dtype=dtype, casting=casting)
    return Tensor(y, device=x1.device)

###
###
###

__all__ = [
    'max', 'maximum'
]