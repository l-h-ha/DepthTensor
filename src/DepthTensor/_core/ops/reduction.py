from operator import index as _integer_index
from typing import Any, Literal

from ...typing import (
    TensorType,
    TensorData,
    TensorDataBool,
    Casting,
    Order,
    DTypeLike,
    Axis,
    TensorLike,
    Device,
    Shape,
)

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
)

from ..utils import (
    _accumulate_operand_grad,
    _as_boolean_mask,
    _cast_gradient_work,
    _check_dtype,
    _check_value_dtype,
    _pointwise_mask,
    _backward_compute_dtype,
    _save_operand_data,
    to_tensordata,
    get_device,
    get_two_operand_op_device,
    NoValue,
)
from ..output import (
    _cuda_pointwise,
    _finish_output,
    _numpy_pointwise,
    _temporary_out,
    _prepare_output,
    _write_output,
)

import numpy as np

try:
    from numpy.exceptions import AxisError
except ImportError:
    from numpy import AxisError

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def _normalize_reduction_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise AxisError(axis, ndim=ndim)

    return axis % ndim

def _prepare_reduction(
    data: TensorData,
    axis: Axis | None,
    keepdims: bool,
    where: TensorDataBool | bool,
    *,
    operation: Literal["sum", "mean", "max"],
    initial: Any = NoValue,
    save_mask: bool = False,
) -> tuple[Axis | None, bool, TensorDataBool | None]:
    ndim = data.ndim
    if isinstance(axis, tuple):
        axis = tuple(_normalize_reduction_axis(_integer_index(part), ndim) for part in axis)
        if len(set(axis)) != len(axis):
            raise ValueError("duplicate value in 'axis'")
    elif axis is not None:
        axis = _integer_index(axis)
        if ndim == 0 and operation != "mean" and axis in (-1, 0):
            axis = ()
        else:
            axis = _normalize_reduction_axis(axis, ndim)

    if not isinstance(keepdims, (bool, np.bool_)):
        keepdims = _integer_index(keepdims)

    mask = None
    if where is not True:
        device = get_device(data)
        xp = np if device == "cpu" else cp
        mask = xp.broadcast_to(_as_boolean_mask(where, device, save=save_mask), data.shape)

    no_identity = (operation == "sum" and initial is None) or (
        operation == "max" and (initial is NoValue or initial is None)
    )
    if no_identity:
        if mask is not None:
            raise ValueError(f"Masked {operation} needs a non-None initial.")

        axes = (axis,) if isinstance(axis, int) else axis
        if axes is None:
            axes = range(ndim)

        if any(data.shape[dim] == 0 for dim in axes):
            name = "add" if operation == "sum" else "maximum"
            raise ValueError(f"zero-size array to reduction operation {name} which has no identity")

    return axis, bool(keepdims), mask

def _reduced_shape(shape: Shape, axis: Axis | None, keepdims: bool) -> Shape:
    if axis is None:
        return (1,) * len(shape) if keepdims else ()

    axes = axis if isinstance(axis, tuple) else (axis,)
    if keepdims:
        return tuple(1 if dim in axes else size for dim, size in enumerate(shape))

    return tuple(size for dim, size in enumerate(shape) if dim not in axes)

def _link_reduction_backward(
    y: TensorType,
    x: TensorLike,
    axis: Axis | None,
    keepdims: bool,
    mask: TensorDataBool | None,
    divisor: int | TensorData | None = None,
) -> None:
    from ...tensor import Tensor

    if not isinstance(x, Tensor) or not x.requires_grad:
        return

    x_saved_version = x._version

    def callback(y_grad: TensorData, x_shape: Shape, device: Device) -> TensorData:
        if device == "cpu":
            xp = np
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            xp = cp

        grad = y_grad
        if divisor is not None:
            grad = _cast_gradient_work(grad, _backward_compute_dtype(grad.dtype, x.dtype)) / divisor

        if x_shape and not keepdims and axis is not None:
            grad = xp.expand_dims(grad, axis)

        grad = xp.broadcast_to(grad, x_shape)

        if mask is not None:
            grad = xp.where(mask, grad, grad.dtype.type(0))

        return grad

    def grad_fn() -> None:
        if not x.requires_grad:
            return

        if x._version != x_saved_version:
            raise RuntimeError(
                "A saved operand was modified by an in-place operation."
            )

        if y.grad is None:
            y.zero_grad()

        grad = callback(y.grad, x.shape, y.device)
        _accumulate_operand_grad(x, grad)

    y.prev = (x,)
    y.grad_fn = grad_fn

def sum(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
    axis: Axis | None = None,
    dtype: DTypeLike | None = None,
    out: TensorData | None = None,
    keepdims: bool = False,
    initial: Any = NoValue,
    where: TensorDataBool | bool = True,
) -> TensorType:
    """
    Sum selected elements along one or more axes.

    Parameters
    ----------
    a : TensorLike
        Input values.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    requires_grad : bool or None, optional
        If None, track gradients when an input does. False disables tracking.
    axis : int or tuple of int, optional
        None reduces all axes.
    dtype : DTypeLike, optional
        Accumulator dtype.
    out : TensorData, optional
        Buffer on the operation's device.
    keepdims : bool, optional
        Keep reduced axes with size one.
    initial : scalar, optional
        Start at zero by default. None requires an unmasked, nonempty slice.
    where : TensorDataBool or bool, optional
        Sum values where the mask is true.

    Returns
    -------
    Tensor
        Sum for each reduced slice.

    Raises
    ------
    ValueError
        Incompatible mask or initial value.
    RuntimeError
        Gradients require a floating-point dtype. `out` cannot overlap a
        tracked tensor, or any input if an input tracks gradients.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor, _check_grad_dtype

    _check_dtype(dtype)
    _check_value_dtype(initial)

    if device is None:
        op_device = get_device(a)
    else:
        op_device = device

    if requires_grad is None:
        requires_grad = a.requires_grad if isinstance(a, Tensor) else False

    if requires_grad and dtype is not None:
        _check_grad_dtype(np.dtype(dtype))

    arr = to_tensordata(a, device=op_device)
    axis, keepdims, mask = _prepare_reduction(
        arr, axis, keepdims, where, operation="sum", initial=initial, save_mask=requires_grad
    )

    out, aliases = _prepare_output(
        out, op_device, a, where, requires_grad=requires_grad,
        shape=_reduced_shape(arr.shape, axis, keepdims) if out is not None else None,
    )

    if op_device == "cpu":
        kwds = {
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": True if mask is None else mask,
        }
        if initial is not NoValue:
            kwds["initial"] = initial
        if out is not None:
            kwds["out"] = _temporary_out(out)

        y = np.sum(arr, **kwds)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        if mask is not None:
            arr = cp.where(mask, arr, arr.dtype.type(0))

        if initial is NoValue or initial is None:
            temp_out = _temporary_out(out)
            y = cp.sum(arr, axis=axis, dtype=dtype, out=temp_out, keepdims=keepdims)
        else:
            acc_dtype = np.dtype(dtype) if dtype is not None else None
            if acc_dtype is None and out is not None:
                acc_dtype = out.dtype
                if "f" in (arr.dtype.kind, out.dtype.kind):
                    acc_dtype = np.result_type(arr.dtype, out.dtype)
            if acc_dtype is None and arr.dtype == np.float16:
                acc_dtype = arr.dtype

            work_dtype = acc_dtype
            if acc_dtype == np.float16:
                arr = arr.astype(acc_dtype, copy=False)
                work_dtype = np.float32

            y = cp.sum(arr, axis=axis, dtype=work_dtype, keepdims=keepdims)
            y = cp.add(y, (acc_dtype or y.dtype).type(initial))

            if out is None and acc_dtype == np.float16:
                y = y.astype(acc_dtype, copy=False)

    y = _write_output(out, y)
    result = _finish_output(y, op_device, requires_grad, aliases)

    if requires_grad and isinstance(a, Tensor) and a.requires_grad:
        _link_reduction_backward(result, a, axis, keepdims, mask)

    return result

def _cuda_max(
    data: TensorData,
    axis: Axis | None,
    keepdims: bool,
    mask: TensorDataBool | None,
    initial: Any,
) -> TensorData:
    if cp is None:
        raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

    if initial is NoValue or initial is None:
        return cp.max(data, axis=axis, keepdims=keepdims)

    lower_bound = data.dtype.type(initial)
    if data.size == 0:
        result = cp.sum(data, axis=axis, dtype=data.dtype, keepdims=keepdims)
        result.fill(lower_bound)
        return result

    if mask is not None:
        data = cp.where(mask, data, lower_bound)

    values = cp.max(data, axis=axis, keepdims=keepdims)
    return cp.maximum(values, lower_bound)

def max(
    a: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
    axis: Axis | None = None,
    out: TensorData | None = None,
    keepdims: bool = False,
    initial: Any = NoValue,
    where: TensorDataBool | bool = True,
) -> TensorType:
    """
    Find the maximum along the given axes. Split gradients evenly among ties.

    Parameters
    ----------
    a : TensorLike
        Input values.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    requires_grad : bool or None, optional
        If None, track gradients when an input does. False disables tracking.
    axis : int or tuple of int, optional
        None reduces all axes.
    out : TensorData, optional
        Buffer on the operation's device.
    keepdims : bool, optional
        Keep reduced axes with size one.
    initial : scalar, optional
        Value to include when finding the maximum. Required with a mask.
    where : TensorDataBool or bool, optional
        Find the maximum where the mask is true.

    Returns
    -------
    Tensor
        Maximum for each reduced slice.

    Raises
    ------
    ValueError
        A mask or empty slice requires initial.
    RuntimeError
        Gradients require a floating-point output dtype. `out` cannot overlap a
        tracked tensor, or any input if an input tracks gradients.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor

    _check_value_dtype(initial)

    if device is None:
        op_device = get_device(a)
    else:
        op_device = device

    if requires_grad is None:
        requires_grad = a.requires_grad if isinstance(a, Tensor) else False

    arr = to_tensordata(a, device=op_device)
    axis, keepdims, mask = _prepare_reduction(arr, axis, keepdims, where, operation="max", initial=initial)

    out, aliases = _prepare_output(
        out, op_device, a, where, requires_grad=requires_grad,
        shape=_reduced_shape(arr.shape, axis, keepdims) if out is not None else None,
    )

    if out is not None and initial is not NoValue and initial is not None:
        arr = arr.astype(np.result_type(arr.dtype, out.dtype), copy=False)

    xp = np if op_device == "cpu" else cp

    if op_device == "cpu":
        kwargs = {"axis": axis, "keepdims": keepdims, "where": True if mask is None else mask}
        if initial is not NoValue:
            kwargs["initial"] = initial

        y = np.max(arr, **kwargs)
    else:
        y = _cuda_max(arr, axis, keepdims, mask, initial)

    tracks_input = requires_grad and isinstance(a, Tensor) and a.requires_grad
    if tracks_input:
        maxima = xp.expand_dims(y, axis) if arr.ndim and not keepdims and axis is not None else y
        max_mask = arr == maxima
        if mask is not None:
            max_mask &= mask
        count = xp.maximum(max_mask.sum(axis=axis, keepdims=keepdims), 1)

    y = _write_output(out, y)
    result = _finish_output(y, op_device, requires_grad, aliases)

    if tracks_input:
        _link_reduction_backward(result, a, axis, keepdims, max_mask, count)

    return result

def _link_maximum_backward(
    y: TensorType, x1: TensorLike, x2: TensorLike, mask: TensorDataBool | None,
    dtype: DTypeLike | None = None,
) -> None:
    from ...tensor import Tensor

    x1_is_tensor = isinstance(x1, Tensor)
    x2_is_tensor = isinstance(x2, Tensor)
    tracks_x1 = x1_is_tensor and x1.requires_grad
    tracks_x2 = x2_is_tensor and x2.requires_grad

    x1_saved_version = x1._version if x1_is_tensor else 0
    x2_saved_version = x2._version if x2_is_tensor else 0
    device = y.device
    x1_data = _save_operand_data(x1, device, dtype)
    x2_data = _save_operand_data(x2, device, dtype)

    def grad_fn() -> None:
        if x1_is_tensor and x1._version != x1_saved_version:
            raise RuntimeError(
                "A saved operand was modified by an in-place operation."
            )

        if x2_is_tensor and x2._version != x2_saved_version:
            raise RuntimeError(
                "A saved operand was modified by an in-place operation."
            )

        if y.grad is None:
            y.zero_grad()

        y_grad = y.grad
        xp = np if device == "cpu" else cp

        # Ties go to x1.
        mask_x1 = x1_data >= x2_data
        mask_x2 = x2_data > x1_data

        if mask is not None:
            mask_x1 = mask_x1 & mask
            mask_x2 = mask_x2 & mask

        if tracks_x1 and x1.requires_grad:
            grad_x1 = xp.where(mask_x1, y_grad, y_grad.dtype.type(0))
            _accumulate_operand_grad(x1, grad_x1)

        if tracks_x2 and x2.requires_grad:
            grad_x2 = xp.where(mask_x2, y_grad, y_grad.dtype.type(0))
            _accumulate_operand_grad(x2, grad_x2)

    prev = []
    if tracks_x1:
        prev.append(x1)
    if tracks_x2:
        prev.append(x2)

    y.prev = tuple(prev)
    y.grad_fn = grad_fn

def maximum(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    """
    Take elementwise maxima. Ties route gradients to x1.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorData, optional
        Buffer on the operation's device.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    requires_grad : bool or None, optional
        If None, track gradients when an input does. False disables tracking.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : DTypeLike, optional
        Operation dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Computed values.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    RuntimeError
        Gradients require a floating-point dtype. `out` cannot overlap a
        tracked tensor, or any input if an input tracks gradients.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor, _check_grad_dtype

    _check_dtype(dtype)

    x1_is_tensor = isinstance(x1, Tensor)
    x2_is_tensor = isinstance(x2, Tensor)
    op_device = get_two_operand_op_device(
        x1, x2, x1_is_tensor=x1_is_tensor, x2_is_tensor=x2_is_tensor, device=device
    )

    if requires_grad is None:
        requires_grad = False
        if x1_is_tensor:
            requires_grad = requires_grad or x1.requires_grad
        if x2_is_tensor:
            requires_grad = requires_grad or x2.requires_grad

    if requires_grad and dtype is not None:
        _check_grad_dtype(np.dtype(dtype))

    out, aliases = _prepare_output(out, op_device, x1, x2, where, requires_grad=requires_grad)

    _x1: TensorData = to_tensordata(x1, device=op_device)
    _x2: TensorData = to_tensordata(x2, device=op_device)
    mask = _pointwise_mask(where, op_device, save=requires_grad)

    if op_device == "cpu":
        y = _numpy_pointwise(
            np.maximum, (_x1, _x2), out, mask, order=order, dtype=dtype, casting=casting, subok=subok
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = _cuda_pointwise(
            cp.maximum, (_x1, _x2), out, mask, order=order, dtype=dtype, casting=casting
        )

    result = _finish_output(y, op_device, requires_grad, aliases)

    if requires_grad:
        _link_maximum_backward(result, x1, x2, mask, dtype)

    return result
