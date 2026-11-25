from __future__ import annotations

from threading import Lock
from weakref import WeakValueDictionary

import numpy as np

from ..typing import TensorData, TensorType

try:
    from numpy.exceptions import TooHardError
except ImportError:
    from numpy import TooHardError

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

_live_tensors: WeakValueDictionary[int, TensorType] = WeakValueDictionary()
_live_tensors_lock = Lock()

def _shares_storage(a: TensorData, b: TensorData) -> bool:
    if a is b:
        return True

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        try:
            return np.shares_memory(a, b, max_work=1_000)
        except TooHardError:
            return True

    if cp is None or not isinstance(a, cp.ndarray) or not isinstance(b, cp.ndarray):
        return False

    if not a.size or not b.size or not cp.may_share_memory(a, b):
        return False

    if a.data.ptr == b.data.ptr or (
        (a.flags.c_contiguous or a.flags.f_contiguous)
        and (b.flags.c_contiguous or b.flags.f_contiguous)
    ):
        return True

    # CuPy compares element addresses. Assume overlap if element sizes or alignment differ.
    if (
        a.itemsize != b.itemsize
        or (a.data.ptr - b.data.ptr) % a.itemsize
        or any(stride % a.itemsize for stride in (*a.strides, *b.strides))
    ):
        return True

    try:
        return cp.shares_memory(a, b)
    except (AttributeError, NotImplementedError, cp.cuda.memory.OutOfMemoryError):
        return True

def _register_tensor(tensor: TensorType) -> None:
    with _live_tensors_lock:
        _live_tensors[id(tensor)] = tensor

def _storage_aliases(data: TensorData) -> tuple[TensorType, ...]:
    """
    Find tensors that may share memory with the input.

    Returns
    -------
    tuple of Tensor
        Live tensors whose arrays may overlap the input.
    """
    with _live_tensors_lock:
        live_tensors = tuple(_live_tensors.values())
    return tuple(tensor for tensor in live_tensors if _shares_storage(tensor.data, data))

def _is_differentiable_result(tensor: TensorType) -> bool:
    if not tensor.requires_grad or (not tensor.prev and tensor.grad_fn is None):
        return False
    from ..tensor import Tensor

    # A view can update its leaf.
    return (
        len(tensor.prev) != 1
        or not isinstance(tensor.prev[0], Tensor)
        or not _shares_storage(tensor.data, tensor.prev[0].data)
    )

def _mark_out_modified(
    aliases: tuple[TensorType, ...],
    result: TensorType | None = None,
    *,
    replaced: TensorType | None = None,
) -> None:
    """
    Mark a write to shared storage.

    Parameters
    ----------
    aliases : tuple of Tensor
        Tensors sharing the modified storage.
    result : Tensor or None, optional
        Result whose version should match the updated array.
    replaced : Tensor or None, optional
        Tensor whose version was already updated.
    """
    replaced_counter = id(replaced._version_counter) if replaced is not None else None
    modified_counters = {}

    for tensor in aliases:
        counter = id(tensor._version_counter)

        if counter != replaced_counter and counter not in modified_counters:
            tensor._version += 1
            modified_counters[counter] = tensor._version_counter

    if result is not None and len(modified_counters) == 1:
        version_counter = next(iter(modified_counters.values()))

        if any(tensor.data is result.data for tensor in aliases):
            result._version_counter = version_counter
        else:
            result._version = version_counter[0]

def _invalidated_result_backward() -> None:
    raise RuntimeError("Differentiable result data was modified by an in-place operation.")

def _mark_external_write(data: TensorData) -> None:
    aliases = _storage_aliases(data)
    _mark_out_modified(aliases)

    for tensor in aliases:
        if _is_differentiable_result(tensor):
            tensor.grad_fn = _invalidated_result_backward
