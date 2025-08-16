from typing import (
    Optional
)

from ...typing import (
    TensorLike, DTypeLike, Order,
    AxisShapeLike
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG
)

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
    dtype: Optional[DTypeLike] = None, 
    order: Order = 'K', 
    subok: bool = True,
    shape: Optional[AxisShapeLike] = None
) -> TensorLike:
    from ...tensor import Tensor
    if a.device == "cpu":
        return Tensor(
            a.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape), 
            device=a.device
        )
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(
            #! As of the time writing this, CuPy does not support subok.
            cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape),
            device=a.device
        )
    
def ones_like(
    a: TensorLike,
    /,
    *, 
    dtype: Optional[DTypeLike] = None, 
    order: Order = 'K', 
    subok: bool = True,
    shape: Optional[AxisShapeLike] = None
) -> TensorLike:
    from ...tensor import Tensor
    if a.is_cpu():
        return Tensor(
            np.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape), 
            device=a.device
        )
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(
            #! As of the time writing this, CuPy does not support subok.
            cp.ones_like(a, dtype=dtype, order=order, subok=None, shape=shape),
            device=a.device
        )
    
###
###
###

__all__ = [
    'zeros_like',
    'ones_like'
]