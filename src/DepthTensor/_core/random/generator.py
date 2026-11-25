from typing import overload, Any
from numpy import random

from ...typing import ArrayLike, DTypeLike, TensorType, Axis, int64, Device, float32
from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
)
from ..utils import _check_device, _check_dtype, _check_value_dtype

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def _cpu_sample_dtype(dtype: DTypeLike | None) -> np.dtype:
    message = "dtype must be float32 or float64."
    try:
        sample_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise RuntimeError(message) from exc

    _check_dtype(sample_dtype)
    if sample_dtype not in (np.float32, np.float64):
        raise RuntimeError(message)

    return sample_dtype

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
    Sample uniformly in `[0, 1)`.

    Parameters
    ----------
    *d : int
        Shape. Omit for a scalar.
    dtype : DTypeLike, optional
        Output dtype. CPU supports float32 and float64.
    device : Device, optional
        Output device.
    requires_grad : bool, optional
        Enable gradient tracking.

    Returns
    -------
    Tensor
        Random samples.

    Notes
    -----
    `depthtensor.random.seed()` controls CPU draws.
    """
    from ...tensor import Tensor

    _check_device(device)

    if device == "cpu":
        sample_dtype = _cpu_sample_dtype(dtype)
        if sample_dtype == np.float32:
            # Use 24 bits to keep float32 draws below 1.
            levels = 1 << 24
            y = random.randint(levels, size=d, dtype=np.uint32).astype(sample_dtype)
            y *= np.float32(1 / levels)
        else:
            y = random.random_sample(size=d)
    else:
        _check_dtype(dtype)

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
    Sample standard normal values.

    Parameters
    ----------
    *d : int
        Shape. Omit for a scalar.
    dtype : DTypeLike, optional
        Output dtype. CPU supports float32 and float64.
    device : Device, optional
        Output device.
    requires_grad : bool, optional
        Enable gradient tracking.

    Returns
    -------
    Tensor
        Random samples.

    Notes
    -----
    `depthtensor.random.seed()` controls CPU draws.
    """
    from ...tensor import Tensor

    _check_device(device)

    if device == "cpu":
        sample_dtype = _cpu_sample_dtype(dtype)
        y = random.standard_normal(size=d).astype(sample_dtype, copy=False)
    else:
        _check_dtype(dtype)

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
) -> TensorType:
    """
    Sample random integers.

    Parameters
    ----------
    low : int
        Inclusive lower bound, or exclusive upper bound if `high` is omitted.
    high : int, optional
        Exclusive upper bound. Omit to sample `[0, low)`.
    size : Axis, optional
        Shape. Omit for a scalar.
    dtype : DTypeLike, optional
        Integer output dtype.
    device : Device, optional
        Output device.

    Returns
    -------
    Tensor
        Integer samples.

    Notes
    -----
    `depthtensor.random.seed()` controls CPU draws.
    """
    from ...tensor import Tensor

    _check_device(device)
    _check_dtype(dtype)
    _check_value_dtype(low)
    _check_value_dtype(high)

    if device == "cpu":
        y = random.randint(low=low, high=high, size=size, dtype=dtype)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = cp.random.randint(low=low, high=high, size=size, dtype=dtype)

    return Tensor(y, requires_grad=False)

def _uniform_float_bounds(low, high, dtype, xp):
    low, high = xp.asarray(low), xp.asarray(high)
    compare_dtype = np.result_type(low.dtype, high.dtype, dtype, np.float64)
    low, high = low.astype(compare_dtype, copy=False), high.astype(compare_dtype, copy=False)

    fixed = low == high
    with np.errstate(over="ignore", under="ignore"):
        minimum = low.astype(dtype)
        maximum = high.astype(dtype)
        minimum = xp.where(
            (minimum < low) & ~fixed, xp.nextafter(minimum, dtype.type(np.inf)), minimum
        )
        maximum = xp.where(
            (maximum >= high) & ~fixed, xp.nextafter(maximum, dtype.type(-np.inf)), maximum
        )

    if xp.any((low > high) | (minimum > maximum)):
        raise ValueError(f"Interval has no values representable in {dtype}.")

    return minimum, maximum

def _scale_wide_uniform_bounds(low, high, xp):
    with np.errstate(over="ignore", invalid="ignore"):
        low = xp.asarray(low, dtype=np.float64)
        high = xp.asarray(high, dtype=np.float64)
        wide = xp.isposinf(high - low) & xp.isfinite(low) & xp.isfinite(high)

    if not xp.any(wide):
        return None

    # Halve intervals whose width overflows float64.
    scale = xp.where(wide, 2.0, 1.0)
    return low / scale, high / scale, scale

def uniform(
    low: ArrayLike = 0.0,
    high: ArrayLike = 1.0,
    size: Axis | None = None,
    *,
    device: Device = "cpu",
    dtype: DTypeLike = float32,
    requires_grad: bool = False,
):
    """
    Draw uniform samples. Equal bounds give a constant.

    Parameters
    ----------
    low : float or array-like, optional
        Inclusive lower bound.
    high : float or array-like, optional
        Exclusive upper bound when the bounds differ.
    size : Axis, optional
        Defaults to the bounds' broadcast shape.
    device : Device, optional
        Output device.
    dtype : DTypeLike, optional
        Output dtype. CPU casts non-floating outputs after sampling.
    requires_grad : bool, optional
        Enable gradient tracking.

    Returns
    -------
    Tensor
        Samples from `[low, high)` for distinct floating-point bounds.

    Raises
    ------
    ValueError
        For floats, low exceeds high or no value in the interval fits the dtype.

    Notes
    -----
    `depthtensor.random.seed()` controls CPU draws.
    """
    from ...tensor import Tensor

    _check_device(device)
    _check_dtype(dtype)
    _check_value_dtype(low)
    _check_value_dtype(high)

    if device == "cpu":
        xp = np
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        xp = cp

    sample_dtype = np.dtype(dtype)
    scale = None
    if sample_dtype.kind == "f":
        # Validate host bounds before drawing on CUDA.
        bounds_xp = np
        if xp is cp and (isinstance(low, cp.ndarray) or isinstance(high, cp.ndarray)):
            bounds_xp = cp

        minimum, maximum = _uniform_float_bounds(low, high, sample_dtype, bounds_xp)
        if xp is not bounds_xp:
            minimum = minimum[()] if minimum.ndim == 0 else xp.asarray(minimum)
            maximum = maximum[()] if maximum.ndim == 0 else xp.asarray(maximum)

        scaled_bounds = _scale_wide_uniform_bounds(low, high, bounds_xp)
        if scaled_bounds is not None:
            low, high, scale = scaled_bounds

    if device == "cpu":
        y = random.uniform(low=low, high=high, size=size)
    else:
        # Draw in float64 when casting to float16 or float32 to avoid width overflow.
        draw_dtype = np.float64 if sample_dtype in (np.float16, np.float32) else sample_dtype
        y = cp.random.uniform(low=low, high=high, size=size, dtype=draw_dtype)

    with np.errstate(over="ignore", under="ignore"):
        if scale is not None:
            y = y * xp.asarray(scale)
        y = xp.asarray(y, dtype=sample_dtype)

    if sample_dtype.kind == "f":
        xp.clip(y, minimum, maximum, out=y)

    return Tensor(y, requires_grad=requires_grad)
