from typing import Optional, overload, Any

from numpy import random

from ...typing import DTypeLike, TensorType, Axis, int64, Device

from ..exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

_RNG = np.random.default_rng()

###
###
###


@overload
def rand(*, device: Device = "cpu", requires_grad: bool = False) -> TensorType: ...
@overload
def rand(
    *d: int,
    dtype: DTypeLike | None = None,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType: ...
def rand(
    *d: int,
    dtype: DTypeLike | None = None,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    from ...tensor import Tensor

    if device == "cpu":
        if dtype is None:
            y = _RNG.random(size=d)
        elif dtype is np.float32 or dtype is np.float64:
            y = _RNG.random(size=d, dtype=dtype)
        else:
            raise RuntimeError(
                "The 'dtype' argument must be a type either a float64 or a float32."
            )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.rand(*d, dtype=dtype)
    return Tensor(y, requires_grad=requires_grad)


@overload
def randn(*, device: Device = "cpu", requires_grad: bool = False) -> TensorType: ...
@overload
def randn(
    *d: int,
    dtype: DTypeLike | None = None,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType: ...


def randn(
    *d: int,
    dtype: DTypeLike | None = None,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    from ...tensor import Tensor

    if device == "cpu":
        if dtype is None:
            y = _RNG.standard_normal(d)
        elif dtype is np.float32 or dtype is np.float64:
            y = _RNG.standard_normal(d, dtype=dtype)
        else:
            raise RuntimeError(
                "The 'dtype' argument must be a type either a float64 or a float32."
            )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.randn(*d, dtype=dtype)
    return Tensor(y, requires_grad=requires_grad)


def randint(
    low: int,
    high: int | None = None,
    size: Axis | None = None,
    dtype: Any = int64,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> TensorType:
    from ...tensor import Tensor

    if device == "cpu":
        y = random.randint(low=low, high=high, size=size, dtype=dtype)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.randint(low=low, high=high, size=size, dtype=dtype)
    return Tensor(y, requires_grad=requires_grad)


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: Axis | None = None,
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
):
    from ...tensor import Tensor

    if device == "cpu":
        y = random.uniform(low=low, high=high, size=size)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.uniform(low=low, high=high, size=size)
    return Tensor(y, requires_grad=requires_grad)


###
###
###

__all__ = ["rand", "randn", "randint", "uniform"]
