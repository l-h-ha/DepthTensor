from operator import index as _integer_index
from typing import Any

import numpy as np

from ..typing import Device, Shape, TensorData, TensorType
from .storage import _is_differentiable_result, _mark_out_modified, _shares_storage, _storage_aliases
from .utils import _check_value_dtype, to_tensordata

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def _prepare_index(index: Any, device: Device, *, save: bool = False) -> Any:
    from ..tensor import Tensor

    def prepare(item: Any) -> Any:
        if isinstance(item, tuple):
            return tuple(prepare(part) for part in item)

        if isinstance(item, list):
            return [prepare(part) for part in item]

        if isinstance(item, Tensor):
            item = item.data if item.device == device else to_tensordata(item.data, device)

        _check_value_dtype(item)
        if not save:
            return item

        if isinstance(item, slice):
            bounds = (item.start, item.stop, item.step)
            return slice(*(None if bound is None else _integer_index(bound) for bound in bounds))

        if isinstance(item, np.ndarray) or (cp is not None and isinstance(item, cp.ndarray)):
            return item.copy()

        if not isinstance(item, (bool, np.bool_)) and hasattr(type(item), "__index__"):
            return _integer_index(item)

        return item

    return prepare(index)

def _cuda_index_component(index: Any) -> tuple[Any, int]:
    if index is None or index is Ellipsis or isinstance(index, (bool, np.bool_)):
        return index, 0

    if isinstance(index, (list, tuple)):
        try:
            index = np.asarray(index)
        except (TypeError, ValueError):
            try:
                index = cp.asarray(index)
            except (TypeError, ValueError):
                return index, 1

    if isinstance(index, (np.ndarray, cp.ndarray)) and index.dtype.kind == "b":
        return index, index.ndim

    return index, 1

def _cuda_index_out_of_bounds(item: Any, size: int) -> bool:
    if isinstance(item, (int, np.integer)):
        return not -size <= int(item) < size

    if not isinstance(item, (np.ndarray, cp.ndarray)) or item.dtype.kind not in "iu" or not item.size:
        return False

    limits = np.iinfo(item.dtype)
    # Bound comparisons must fit the index dtype.
    invalid = item >= size if size <= limits.max else None
    if item.dtype.kind == "i" and -size > limits.min:
        too_negative = item < -size
        invalid = too_negative if invalid is None else invalid | too_negative

    return invalid is not None and bool(invalid.any())

def _check_cuda_index_bounds(index: Any, shape: Shape) -> None:
    """
    Reject indices that CuPy would wrap.
    """
    entries = index if isinstance(index, tuple) else (index,)
    components = [_cuda_index_component(item) for item in entries]
    ellipsis_width = len(shape) - sum(consumed for _, consumed in components)

    if ellipsis_width < 0:
        return

    axis = 0
    for item, consumed in components:
        if item is Ellipsis:
            axis += ellipsis_width
            continue

        if not consumed:
            continue

        if axis >= len(shape):
            return

        if _cuda_index_out_of_bounds(item, shape[axis]):
            raise IndexError(f"index is out of bounds for axis {axis} with size {shape[axis]}")

        axis += consumed

def _indexed_cpu_aliases(data: np.ndarray, index: Any, selected: Any) -> tuple[TensorType, ...]:
    parts = index if isinstance(index, tuple) else (index,)
    if len(parts) == data.ndim and all(
        isinstance(part, (int, np.integer)) and not isinstance(part, bool) for part in parts
    ):
        selected = data[(*parts, Ellipsis)]

    if isinstance(selected, np.ndarray):
        if not selected.size and (data.dtype.kind != "O" or not data.ndim):
            return ()

        if _shares_storage(data, selected):
            return _storage_aliases(selected)

    aliases = _storage_aliases(data)
    if not data.ndim:
        return aliases

    affected = [tensor for tensor in aliases if tensor.data is data]
    remaining = {id(tensor): tensor for tensor in aliases if tensor.data is not data}

    if not remaining and data.dtype.kind != "O":
        return tuple(affected)

    # Advanced indexing returns a copy. Use the indices to find affected views.
    coordinates = []
    for axis, size in enumerate(data.shape):
        shape = (1,) * axis + (size,) + (1,) * (data.ndim - axis - 1)
        grid = np.broadcast_to(np.arange(size).reshape(shape), data.shape)
        coordinates.append(np.asarray(grid[index]).flat)

    if not len(coordinates[0]):
        return ()

    for coordinate in zip(*coordinates):
        view = data[(*coordinate, Ellipsis)]
        for key, tensor in tuple(remaining.items()):
            if _shares_storage(view, tensor.data):
                affected.append(remaining.pop(key))

        if not remaining:
            break

    return tuple(affected)

def _write_index(data: TensorData, index: Any, value: Any, device: Device) -> None:
    from ..tensor import Tensor

    index = _prepare_index(index, device)
    if device == "cuda":
        _check_cuda_index_bounds(index, data.shape)

    selected = data[index] if device == "cpu" else None
    aliases = _indexed_cpu_aliases(data, index, selected) if device == "cpu" else _storage_aliases(data)
    if any(_is_differentiable_result(tensor) for tensor in aliases):
        raise RuntimeError("In-place indexed assignment would overwrite a differentiable result.")

    if isinstance(value, Tensor):
        value = to_tensordata(value, device)
    else:
        _check_value_dtype(value)

    if device == "cpu" and data.dtype.kind != "O" and (
        data.dtype.fields is not None
        or not isinstance(value, (np.ndarray, np.generic))
        or value.dtype != data.dtype
    ):
        if isinstance(selected, np.ndarray) or data.dtype.fields is not None:
            # Finish the cast before writing to the array.
            value = np.asarray(value, dtype=selected.dtype)

    data[index] = value
    _mark_out_modified(aliases)
