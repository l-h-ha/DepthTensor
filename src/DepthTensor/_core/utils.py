from contextvars import ContextVar
from typing import Any
from ..typing import Device, DTypeLike, TensorLike, TensorData, TensorDataBool, TensorType

from .exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

import numpy as np

_backward_active = ContextVar("depthtensor_backward_active", default=False)

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None
    _cuda_widen_float32 = None
else:
    # CuPy's cast can flush float32 subnormals to zero.
    _cuda_widen_float32 = cp.ElementwiseKernel(
        "float32 x", "float64 y",
        'asm("cvt.f64.f32 %0, %1;" : "=d"(y) : "f"(x));',
        "depthtensor_widen_float32",
    )

def _check_device(device: Device) -> None:
    if not isinstance(device, str) or device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")

def _check_dtype(dtype: DTypeLike | None) -> None:
    if dtype is None:
        return

    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        raise RuntimeError("Complex dtypes are unsupported.")

    if dtype.subdtype is not None:
        _check_dtype(dtype.subdtype[0])

    if dtype.names is not None:
        for name in dtype.names:
            _check_dtype(dtype.fields[name][0])

def _check_value_dtype(value: Any, seen: set[int] | None = None) -> None:
    if isinstance(value, complex):
        raise RuntimeError("Complex dtypes are unsupported.")

    if isinstance(value, (list, tuple)):
        values = value
    else:
        _check_dtype(getattr(value, "dtype", None))
        if not isinstance(value, np.ndarray) or value.dtype.kind != "O":
            return
        values = value.flat

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    for item in values:
        _check_value_dtype(item, seen)

def tensordata_to_device(obj: TensorData, device: Device) -> TensorData:
    """
    Move array data to a device.

    Parameters
    ----------
    obj : TensorData
        NumPy or CuPy array, or NumPy scalar.
    device : Device
        Target device.

    Returns
    -------
    TensorData
        Array on the target device.

    Raises
    ------
    CuPyNotFound
        CUDA requested without CuPy.
    RuntimeError
        Unsupported obj type.
    """
    _check_device(device)
    _check_value_dtype(obj)

    if isinstance(obj, np.generic):
        obj = np.asarray(obj)

    if isinstance(obj, np.ndarray):
        if device == "cpu":
            return obj
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            return cp.array(obj)
    else:
        if cp is not None and isinstance(obj, cp.ndarray):
            if device == "cuda":
                return obj
            return cp.asnumpy(obj)
        else:
            raise RuntimeError(
                f"Expected NumPy or CuPy data, got {type(obj)}."
            )

def _backward_compute_dtype(*dtypes: DTypeLike) -> np.dtype:
    return np.result_type(np.float32, *dtypes)

def _cast_gradient_work(data: TensorData, dtype: np.dtype, *, copy: bool = False) -> TensorData:
    if data.dtype == np.float32 and dtype == np.float64 and (
        cp is not None and isinstance(data, cp.ndarray)
    ):
        return _cuda_widen_float32(data)

    return data.astype(dtype, copy=copy)

def _cast_backward_inputs(
    grad: TensorData, dtype: DTypeLike, *operands: TensorData | None
) -> tuple[TensorData, tuple[TensorData | None, ...]]:
    # Keep the saved forward values when widening them for backward.
    work_dtype = _backward_compute_dtype(
        grad.dtype, dtype,
        *(data.dtype for data in operands if data is not None and data.dtype.kind == "f"),
    )
    values = tuple(
        _cast_gradient_work(data, work_dtype)
        if data is not None and data.dtype.kind == "f" else data
        for data in operands
    )

    return _cast_gradient_work(grad, work_dtype), values

def unbroadcast_tensordata_to_shape(
    result: TensorData, target_shape: tuple, device: Device, *, dtype: DTypeLike | None = None
) -> Any:
    """
    Sum over axes added or expanded by broadcasting.

    Parameters
    ----------
    result : TensorData
        Broadcast gradient.
    target_shape : tuple
        Shape before broadcasting.
    device : Device
        Device for the sums.
    dtype : DTypeLike, optional
        Minimum precision for the sums.

    Returns
    -------
    TensorData
        Gradient with target_shape.
    """

    if result.shape == target_shape:
        return result

    added_dims = result.ndim - len(target_shape)
    expanded_axes = tuple(
        axis for axis, size in enumerate(target_shape, start=added_dims)
        if size == 1 and result.shape[axis] != 1
    )
    axes = (*range(added_dims), *expanded_axes)

    xp = np if device == "cpu" else cp
    if xp is None:
        raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

    if dtype is not None:
        dtype = _backward_compute_dtype(result.dtype, dtype)
        if xp is cp and result.dtype == np.float32 and dtype == np.float64:
            result = _cast_gradient_work(result, dtype)

    result = xp.sum(result, axis=axes, dtype=dtype, keepdims=True)
    return result.reshape(result.shape[added_dims:])

def to_tensordata(
    a: TensorLike, device: Device | None = None, *, dtype: DTypeLike | None = None
) -> TensorData:
    """
    Convert input to array data.

    Parameters
    ----------
    a : TensorLike
        Input values or Tensor.
    device : Device or None, optional
        Target device. None infers it from a.
    dtype : DTypeLike or None, optional
        Target dtype. None infers it from a.

    Returns
    -------
    TensorData
        Array on the target device.
    """
    from ..tensor import Tensor

    _check_dtype(dtype)

    if device is None:
        device = get_device(a)
    _check_device(device)

    if isinstance(a, Tensor):
        y = a.data
    elif isinstance(a, np.ndarray) or (cp is not None and isinstance(a, cp.ndarray)):
        y = a
    else:
        if dtype is not None:
            _check_value_dtype(a)

        if device == "cpu":
            y = np.array(a, dtype=dtype)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            y = cp.array(a, dtype=dtype)

    _check_value_dtype(y)

    if dtype is not None and dtype != y.dtype:
        y = y.astype(dtype)

    if get_device(y) != device:
        return tensordata_to_device(y, device)

    return y

def _gradient_accumulator(operand: TensorType, contribution: TensorData) -> TensorData:
    current = operand.grad
    if not _backward_active.get():
        if current is None:
            operand.zero_grad()
        return operand.grad

    dtype = _backward_compute_dtype(
        operand.dtype, contribution.dtype, current.dtype if current is not None else operand.dtype
    )
    if current is None:
        xp = np if operand.device == "cpu" else cp
        if xp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        operand.grad = xp.zeros_like(operand.data, dtype=dtype)
    elif current.dtype != dtype:
        operand.grad = _cast_gradient_work(current, dtype)

    return operand.grad

def _accumulate_operand_grad(operand: TensorType, grad: TensorData) -> None:
    if grad.shape != operand.shape:
        grad = unbroadcast_tensordata_to_shape(
            grad, operand.shape, get_device(grad), dtype=operand.dtype
        )

    grad = to_tensordata(grad, device=operand.device)

    accumulator = _gradient_accumulator(operand, grad)
    if _backward_active.get():
        grad = _cast_gradient_work(grad, accumulator.dtype)

    accumulator += grad

def _cast_compute_operand(data: TensorData, dtype: Any) -> TensorData:
    if dtype is None:
        return data

    return data.astype(dtype, copy=False)

def _save_operand_data(
    operand: TensorLike, device: Device, dtype: DTypeLike | None
) -> TensorData:
    """
    Save forward values for backward.

    Parameters
    ----------
    operand : TensorLike
        Input to save.
    device : Device
        Operation device.
    dtype : DTypeLike or None
        Compute dtype. None keeps the input dtype.

    Returns
    -------
    TensorData
        Independent copy.
    """
    return _cast_compute_operand(to_tensordata(operand, device), dtype).copy()

def _as_boolean_mask(
    value: TensorLike, device: Device, *, save: bool, require_bool: bool = False
) -> TensorDataBool:
    from ..tensor import Tensor

    mask = to_tensordata(value, device)
    if mask.dtype.kind != "b":
        if require_bool:
            raise TypeError("where must be boolean.")
        return mask.astype(bool)

    source = value.data if isinstance(value, Tensor) else value
    return mask.copy() if save and mask is source else mask

def _pointwise_mask(
    where: TensorDataBool | bool, device: Device, *, save: bool
) -> TensorDataBool | None:
    if where is True:
        return None

    return _as_boolean_mask(where, device, save=save, require_bool=True)

def get_device(a: TensorLike) -> Device:
    """
    Infer the device of an input.

    Parameters
    ----------
    a : TensorLike
        Input data.

    Returns
    -------
    Device
        Either "cpu" or "cuda".

    Raises
    ------
    RuntimeError
        If a has an unsupported type.
    """
    from ..tensor import Tensor

    if isinstance(a, Tensor):
        return a.device
    elif isinstance(a, (np.ndarray, np.number, np.bool_)):
        return "cpu"
    elif cp is not None and isinstance(a, (cp.ndarray, cp.number, cp.bool_)):
        return "cuda"
    elif isinstance(a, complex):
        raise RuntimeError("Complex dtypes are unsupported.")
    elif isinstance(a, (int, float, list, tuple, bool)):
        return "cpu"
    else:
        raise RuntimeError(f"Invalid argument type: {type(a)}")

def _is_scalar_or_sequence(operand: TensorLike) -> bool:
    return isinstance(operand, (int, float, np.number, np.bool_, list, tuple))

def get_two_operand_op_device(
    x1: TensorLike,
    x2: TensorLike,
    x1_is_tensor: bool,
    x2_is_tensor: bool,
    device: Device | None,
) -> Device:
    if device is not None:
        return device

    if x1_is_tensor and x2_is_tensor:
        if x1.device != x2.device:
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)
        op_device = x1.device
    else:
        if x1_is_tensor:
            op_device = x1.device
            if get_device(x2) != x1.device and not _is_scalar_or_sequence(x2):
                raise DeviceMismatch(
                    f"Expected {x1.device} operand, got {type(x2)}."
                )
        elif x2_is_tensor:
            op_device = x2.device
            if get_device(x1) != x2.device and not _is_scalar_or_sequence(x1):
                raise DeviceMismatch(
                    f"Expected {x2.device} operand, got {type(x1)}."
                )
        else:
            if isinstance(x1, np.ndarray) or isinstance(x2, np.ndarray):
                return "cpu"

            if cp is not None and (
                isinstance(x1, cp.ndarray) or isinstance(x2, cp.ndarray)
            ):
                return "cuda"

            return get_device(x1)

    return op_device

NoValue = object()
