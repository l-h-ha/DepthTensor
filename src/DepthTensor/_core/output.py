from __future__ import annotations

import numpy as np

from ..typing import Casting, Device, DTypeLike, Order, Shape, TensorData, TensorDataBool, TensorLike, TensorType
from .storage import _mark_out_modified, _shares_storage, _storage_aliases
from .utils import _check_value_dtype, get_device

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def _prepare_output(
    out: TensorData | None,
    device: Device,
    *operands: TensorLike,
    requires_grad: bool = False,
    shape: Shape | None = None,
) -> tuple[TensorData | None, tuple[TensorType, ...]]:
    if isinstance(out, tuple):
        if len(out) != 1:
            raise ValueError("The out tuple must contain one item.")
        out = out[0]

    if out is None:
        return None, ()

    if not isinstance(out, np.ndarray) and (cp is None or not isinstance(out, cp.ndarray)):
        raise TypeError("out must be an array.")

    _check_value_dtype(out)

    if get_device(out) != device:
        raise ValueError(f"out buffer must be on the {device} device.")

    if not getattr(out.flags, "writeable", True):
        raise ValueError("out buffer must be writable.")

    if shape is not None and out.shape != shape:
        raise ValueError("out shape must match the result shape.")

    from ..tensor import Tensor, _check_grad_dtype

    tracked_input = any(isinstance(operand, Tensor) and operand.requires_grad for operand in operands)
    if tracked_input and any(
        _shares_storage(operand.data if isinstance(operand, Tensor) else operand, out)
        for operand in operands
    ):
        raise RuntimeError("out buffer overlaps an input in a differentiable operation.")

    aliases = _storage_aliases(out)
    if any(tensor.requires_grad for tensor in aliases):
        raise RuntimeError("out buffer overlaps a differentiable Tensor.")

    if requires_grad:
        _check_grad_dtype(out.dtype)

    return out, aliases

def _check_pointwise_output_shape(out: TensorData | None, *inputs: TensorData | None) -> None:
    if out is None:
        return

    shapes = (value.shape for value in inputs if value is not None)
    if np.broadcast_shapes(out.shape, *shapes) != out.shape:
        raise ValueError("Input and mask shapes must broadcast to out.shape.")

def _check_output_cast(source: DTypeLike, target: DTypeLike, casting: Casting) -> None:
    if not np.can_cast(source, target, casting=casting):
        raise TypeError(f"Cannot cast output from {source} to {target} with casting={casting!r}.")

def _temporary_out(
    out: TensorData | None,
    dtype: DTypeLike | None = None,
    *,
    casting: Casting = "unsafe",
) -> TensorData | None:
    if out is None:
        return None

    dtype = out.dtype if dtype is None else np.dtype(dtype)
    _check_output_cast(dtype, out.dtype, casting)
    xp = np if isinstance(out, np.ndarray) else cp
    return xp.empty_like(out, dtype=dtype)

def _write_output(
    out: TensorData | None,
    values: TensorData,
    *,
    casting: Casting = "unsafe",
    where: TensorDataBool | None = None,
) -> TensorData:
    if out is None or values is out:
        return values

    _check_output_cast(values.dtype, out.dtype, casting)
    xp = np if isinstance(out, np.ndarray) else cp

    if where is not None and _shares_storage(out, where):
        where = where.copy()
    kwargs = {} if where is None else {"where": where}

    if xp is np and values.dtype != out.dtype:
        # Cast into a temporary array so a failed cast cannot partly update out.
        converted = np.empty_like(out)
        np.copyto(converted, values, casting=casting, **kwargs)
        values = converted

    xp.copyto(out, values, casting=casting, **kwargs)
    return out

def _finish_output(
    data: TensorData,
    device: Device,
    requires_grad: bool,
    aliases: tuple[TensorType, ...],
    *,
    destination: TensorType | None = None,
) -> TensorType:
    if destination is not None:
        destination._replace_data(data, aliases=aliases)
        return destination

    from ..tensor import Tensor

    result = Tensor._fast_init(data, device=device, requires_grad=requires_grad)
    _mark_out_modified(aliases, result)
    return result

def _numpy_pointwise(func, inputs, out, mask, *, supports_where=True, **kwargs) -> TensorData:
    if supports_where:
        _check_pointwise_output_shape(out, *inputs, mask)
        kwargs["where"] = True if mask is None else mask

    values = func(*inputs, out=_temporary_out(out), **kwargs)
    return _write_output(out, values, where=mask)

def _cuda_pointwise(
    func, inputs, out, mask, *, order: Order | None = "K", work_dtype=None,
    output_casting: Casting = "unsafe", supports_where=True, **kwargs,
) -> TensorData:
    if order is None:
        order = "K"
    elif isinstance(order, str):
        order = order.upper()
    else:
        raise TypeError("order must be a string or None")

    if order not in ("C", "F", "A", "K"):
        raise ValueError("order must be one of 'C', 'F', 'A', or 'K'")
    if order == "A":
        order = "F" if all(value.flags.f_contiguous for value in inputs if value is not None) else "C"

    if supports_where:
        _check_pointwise_output_shape(out, *inputs, mask)

    temp_out = _temporary_out(out, work_dtype, casting=output_casting)
    if mask is None:
        values = func(*inputs, out=temp_out, **kwargs)
        if out is None and (
            (order == "C" and not values.flags.c_contiguous)
            or (order == "F" and not values.flags.f_contiguous)
        ):
            return values.copy(order=order)
        return _write_output(out, values, casting=output_casting)

    # Replace masked-out values with 1 before computing.
    inputs = tuple(
        cp.where(mask, value, value.dtype.type(1)) if value is not None else None
        for value in inputs
    )

    if temp_out is None:
        values = func(*inputs, **kwargs)
        out = cp.empty_like(values, order=order)
    else:
        values = func(*inputs, out=temp_out, **kwargs)

    return _write_output(out, values, casting=output_casting, where=mask)
