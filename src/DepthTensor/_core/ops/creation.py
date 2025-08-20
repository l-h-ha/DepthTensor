from typing import (
    Optional
)

from ...typing import (
    TensorLike, 
    DTypeLike, 
    Order,
    AxisShapeLike,
    OperandLike,
    DeviceLike
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG
)

from ..utils import to_xp_array

import numpy as np
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###

def zeros_like(
    a: OperandLike,
    /,
    *,
    device: DeviceLike = "cpu",
    dtype: Optional[DTypeLike] = None, 
    order: Order = 'K', 
    subok: bool = True,
    shape: Optional[AxisShapeLike] = None
) -> TensorLike:
    from ...tensor import Tensor
    if isinstance(a, Tensor):
        device_op = a.device
    else:
        device_op = device
    a = to_xp_array(a)
    if device_op == "cpu":
        return Tensor(
            np.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape), 
            device=device_op
        )
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(
            #! As of the time writing this, CuPy does not support subok.
            cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape),
            device=device_op
        )
    
def ones_like(
    a: OperandLike,
    /,
    *,
    device: DeviceLike = "cpu",
    dtype: Optional[DTypeLike] = None, 
    order: Order = 'K', 
    subok: bool = True,
    shape: Optional[AxisShapeLike] = None
) -> TensorLike:
    from ...tensor import Tensor
    if isinstance(a, Tensor):
        device_op = a.device
    else:
        device_op = device
    a = to_xp_array(a)
    if device_op == "cpu":
        return Tensor(
            np.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape), 
            device=device_op
        )
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(
            #! As of the time writing this, CuPy does not support subok.
            cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape),
            device=device_op
        )
    
###
###
###

__all__ = [
    'zeros_like',
    'ones_like'
]