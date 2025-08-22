from typing import (
    Optional,
    overload,
    Any
)

from numpy import random

from ...typing import (
    DTypeLike,
    TensorLike,
    AxisLike,
    int64,
    DeviceLike
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG
)

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###

@overload
def rand(*, device: DeviceLike = "cpu") -> TensorLike: ...
@overload
def rand(
    *d: int,
    dtype: Optional[DTypeLike] = None,
    device: DeviceLike = "cpu"
) -> TensorLike: ... 
def rand(
    *d: int,
    dtype: Optional[DTypeLike] = None,
    device: DeviceLike = "cpu"
) -> TensorLike:
    from ...tensor import Tensor
    if device == "cpu":
        y = random.rand(*d)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.rand(*d, dtype=dtype)
    return Tensor(y, dtype=dtype)
    
@overload
def randn(*, device: DeviceLike = "cpu") -> TensorLike: ...
@overload
def randn(
    *d: int,
    dtype: Optional[DTypeLike] = None,
    device: DeviceLike = "cpu"
) -> TensorLike: ... 
def randn(
    *d: int,
    dtype: Optional[DTypeLike] = None,
    device: DeviceLike = "cpu"
) -> TensorLike:
    from ...tensor import Tensor
    if device == "cpu":
        y = random.randn(*d)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.randn(*d, dtype=dtype)
    return Tensor(y, dtype=dtype)
    
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

def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[AxisLike] = None,
    *,
    device: DeviceLike = "cpu"
):
    from ...tensor import Tensor
    if device == "cpu":
        y = random.uniform(low=low, high=high, size=size)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.uniform(low=low, high=high, size=size)
    return Tensor(y)

###
###
###

__all__ = [
    'rand', 
    'randn', 
    'randint', 
    'uniform'
]