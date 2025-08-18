from typing import (
    Optional,
    overload,
    Any,
    Type
)

from numpy import random

from ...typing import (
    DTypeLike,
    TensorLike,
    AxisLike,
    int64,
    DeviceLike,
    floating
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

@overload
def rand() -> TensorLike: ...
@overload
def rand(
    *d: int,
    dtype: Optional[DTypeLike] = None
) -> TensorLike: ... 
def rand(
    *d: int,
    dtype: Optional[DTypeLike] = None
) -> TensorLike:
    from ...tensor import Tensor
    if dtype is None:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(cp.random.rand(*d, dtype=dtype))
    else:
        return Tensor(np.random.rand(*d))
    
@overload
def randn() -> TensorLike: ...
@overload
def randn(
    *d: int,
    dtype: Optional[DTypeLike] = None
) -> TensorLike: ... 
def randn(
    *d: int,
    dtype: Optional[DTypeLike] = None
) -> TensorLike:
    from ...tensor import Tensor
    if dtype is None:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(cp.random.randn(*d, dtype=dtype))
    else:
        return Tensor(random.randn(*d))
    
def randint(
    low: int,
    high: Optional[int] = None,
    size: Optional[AxisLike] = None,
    dtype: Any = int64,
    device: DeviceLike = "cpu"
) -> TensorLike:
    from ...tensor import Tensor
    if device == "cpu":
        y = random.randint(low=low, high=high, size=size, dtype=dtype)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.randint(low=low, high=high, size=size, dtype=dtype)
    return Tensor(y)

@overload
def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[AxisLike] = None
): ...
@overload
def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[AxisLike] = None,
    dtype: Optional[Type[float]] = None,
): ...
def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[AxisLike] = None,
    dtype: Optional[Type[float]] = None,
):
    from ...tensor import Tensor
    if dtype is None:
        y = random.uniform(low=low, high=high, size=size)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.uniform(low=low, high=high, size=size, dtype=dtype)
    return Tensor(y)

###
###
###

__all__ = [
    'rand', 'randn', 'randint', 'uniform'
]