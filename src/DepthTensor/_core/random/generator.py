from typing import Optional, overload, Any

from numpy import random

from ...typing import DTypeLike, TensorType, Axis, int64, Device, float32

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
    """
    Random values in a given shape.

    Create an array of the given shape and populate it with
    random samples from a uniform distribution over [0, 1).

    Parameters
    ----------
    *d : int
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.
    dtype : DTypeLike | None, optional
        The desired data-type for the array.
    device : Device, optional
        The device to place the result on. Default is 'cpu'.
    requires_grad : bool, optional
        Whether the result requires gradient computation.

    Returns
    -------
    TensorType
        Random values.
    """
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
    """
    Return a sample (or samples) from the "standard normal" distribution.

    Parameters
    ----------
    *d : int
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.
    dtype : DTypeLike | None, optional
        The desired data-type for the array.
    device : Device, optional
        The device to place the result on. Default is 'cpu'.
    requires_grad : bool, optional
        Whether the result requires gradient computation.

    Returns
    -------
    TensorType
        A floating-point array of shape `d` of drawn samples, or a single
        sample if `d` is not provided.
    """
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
    """
    Return random integers from `low` (inclusive) to `high` (exclusive).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        `high=None`, in which case this parameter is one above the
        highest such integer).
    high : int | None, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if `high=None`).
    size : Axis | None, optional
        Output shape. If the given shape is, e.g., `(m, n, k)`, then
        `m * n * k` samples are drawn. Default is None, in which case a
        single value is returned.
    dtype : Any, optional
        Desired dtype of the result.
    device : Device, optional
        The device to place the result on. Default is 'cpu'.
    requires_grad : bool, optional
        Whether the result requires gradient computation.

    Returns
    -------
    TensorType
        size-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` is not provided.
    """
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
    dtype: DTypeLike = float32,
    requires_grad: bool = False,
):
    """
    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    `[low, high)` (includes low, but excludes high).

    Parameters
    ----------
    low : float, optional
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low. The default value is 0.0.
    high : float, optional
        Upper boundary of the output interval. All values generated will be
        less than high. The default value is 1.0.
    size : Axis | None, optional
        Output shape. If the given shape is, e.g., `(m, n, k)`, then
        `m * n * k` samples are drawn.
    device : Device, optional
        The device to place the result on. Default is 'cpu'.
    dtype : DTypeLike, optional
        Desired dtype of the result.
    requires_grad : bool, optional
        Whether the result requires gradient computation.

    Returns
    -------
    TensorType
        Drawn samples from the parameterized uniform distribution.
    """
    from ...tensor import Tensor

    if device == "cpu":
        y = random.uniform(low=low, high=high, size=size).astype(dtype)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.random.uniform(low=low, high=high, size=size, dtype=dtype)
    return Tensor(y, requires_grad=requires_grad)


###
###
###

__all__ = ["rand", "randn", "randint", "uniform"]
