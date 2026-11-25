from typing import overload, Union, Tuple

from ...typing import (
    TensorType,
    Device,
    TensorDataBool,
    Casting,
    Order,
    TensorLike,
)

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

from ..utils import (
    _accumulate_operand_grad,
    _as_boolean_mask,
    _check_dtype,
    _is_scalar_or_sequence,
    _pointwise_mask,
    to_tensordata,
    get_device,
    get_two_operand_op_device,
)
from ..output import _cuda_pointwise, _finish_output, _numpy_pointwise, _prepare_output

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def _link_where_backward(
    result: TensorType, x: TensorLike, y: TensorLike, condition: TensorDataBool
) -> None:
    from ...tensor import Tensor

    branches = tuple(
        (branch, selected_when_true, branch._version)
        for branch, selected_when_true in ((x, True), (y, False))
        if isinstance(branch, Tensor) and branch.requires_grad
    )

    if not branches:
        return

    xp = np if result.device == "cpu" else cp

    def grad_fn() -> None:
        for branch, _, version in branches:
            if branch._version != version:
                raise RuntimeError(
                    "A saved operand was modified by an in-place operation."
                )

        if result.grad is None:
            result.zero_grad()

        grad = result.grad
        zero = grad.dtype.type(0)

        for branch, selected_when_true, _ in branches:
            if not branch.requires_grad:
                continue

            if selected_when_true:
                selected = xp.where(condition, grad, zero)
            else:
                selected = xp.where(condition, zero, grad)

            _accumulate_operand_grad(branch, selected)

    result.prev = tuple(branch for branch, _, _ in branches)
    result.grad_fn = grad_fn

@overload
def where(
    condition: TensorLike,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
) -> Tuple[TensorType, ...]: ...

@overload
def where(
    condition: TensorLike,
    x: TensorLike | None,
    y: TensorLike | None,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool = False,
) -> TensorType: ...

def where(
    condition: TensorLike,
    x: TensorLike | None = None,
    y: TensorLike | None = None,
    /,
    *,
    device: Device | None = None,
    requires_grad: bool | None = None,
) -> Union[tuple[TensorType, ...], TensorType]:
    """
    Select x where condition is true and y elsewhere.

    Parameters
    ----------
    condition : TensorLike
        Broadcastable condition.
    x : TensorLike, optional
        Values where condition is true.
    y : TensorLike, optional
        Values where condition is false. Supply both branches or neither.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    requires_grad : bool or None, optional
        If None, track gradients when x or y does. False disables tracking.

    Returns
    -------
    Tensor or tuple of Tensor
        Selected values, or index tensors per axis without branches.

    Raises
    ------
    ValueError
        Only one branch supplied.
    DeviceMismatch
        Inputs use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    from ...tensor import Tensor

    if (x is None) and (y is None):
        if requires_grad is None:
            requires_grad = False

        if device is None:
            device = get_device(condition)

        data = to_tensordata(condition, device=device)

        if device == "cpu":
            result = np.where(data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            result = cp.where(data)

        return tuple(
            [
                Tensor._fast_init(array, device=device, requires_grad=requires_grad)
                for array in result
            ]
        )
    elif x is not None and y is not None:
        tracks_branches = any(isinstance(branch, Tensor) and branch.requires_grad for branch in (x, y))
        if requires_grad is None:
            requires_grad = tracks_branches

        if device is None:
            devices = {
                get_device(operand)
                for operand in (condition, x, y)
                if not _is_scalar_or_sequence(operand)
            }

            if len(devices) > 1:
                raise DeviceMismatch(DEVICE_MISMATCH_MSG)
            device = next(iter(devices), "cpu")

        data = (
            _as_boolean_mask(condition, device, save=tracks_branches)
            if requires_grad else to_tensordata(condition, device)
        )
        x_data = to_tensordata(x, device=device)
        y_data = to_tensordata(y, device=device)

        if device == "cpu":
            result = np.where(data, x_data, y_data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            result = cp.where(data, x_data, y_data)

        selected = Tensor._fast_init(result, device=device, requires_grad=requires_grad)
        if requires_grad and tracks_branches:
            _link_where_backward(selected, x, y, data)

        return selected
    else:
        raise ValueError("Supply both x and y.")

def _compare(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    func_name: str,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    from ...tensor import Tensor

    _check_dtype(dtype)

    x1_is_tensor, x2_is_tensor = isinstance(x1, Tensor), isinstance(x2, Tensor)
    op_device = get_two_operand_op_device(x1, x2, x1_is_tensor, x2_is_tensor, device)

    out, aliases = _prepare_output(out, op_device, x1, x2, where)

    x1 = to_tensordata(x1, device=op_device)
    x2 = to_tensordata(x2, device=op_device)
    mask = _pointwise_mask(where, op_device, save=False)

    if op_device == "cpu":
        y = _numpy_pointwise(
            getattr(np, func_name), (x1, x2), out, mask,
            order=order, dtype=dtype, casting=casting, subok=subok,
        )
    elif op_device == "cuda":
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = _cuda_pointwise(
            getattr(cp, func_name), (x1, x2), out, mask, order=order, dtype=dtype, casting=casting
        )

    return _finish_output(y, op_device, False, aliases)

def equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    """
    Test elementwise equality.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorDataBool, optional
        Boolean output buffer.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : None, optional
        Default Boolean dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Boolean result.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    return _compare(
        x1,
        x2,
        out=out,
        func_name="equal",
        device=device,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )

def not_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    """
    Test elementwise inequality.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorDataBool, optional
        Boolean output buffer.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : None, optional
        Default Boolean dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Boolean result.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    return _compare(
        x1,
        x2,
        out=out,
        func_name="not_equal",
        device=device,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )

def greater(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    """
    Test x1 > x2 elementwise.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorDataBool, optional
        Boolean output buffer.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : None, optional
        Default Boolean dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Boolean result.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    return _compare(
        x1,
        x2,
        out=out,
        func_name="greater",
        device=device,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )

def greater_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    """
    Test x1 >= x2 elementwise.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorDataBool, optional
        Boolean output buffer.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : None, optional
        Default Boolean dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Boolean result.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    return _compare(
        x1,
        x2,
        out=out,
        func_name="greater_equal",
        device=device,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )

def less(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    """
    Test x1 < x2 elementwise.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorDataBool, optional
        Boolean output buffer.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : None, optional
        Default Boolean dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Boolean result.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    return _compare(
        x1,
        x2,
        out=out,
        func_name="less",
        device=device,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )

def less_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorDataBool | None = None,
    *,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorType:
    """
    Test x1 <= x2 elementwise.

    Parameters
    ----------
    x1 : TensorLike
        First broadcastable operand.
    x2 : TensorLike
        Second broadcastable operand.
    out : TensorDataBool, optional
        Boolean output buffer.
    device : {'cpu', 'cuda'}, optional
        If None, choose the device from the inputs.
    where : TensorDataBool or bool, optional
        Write where the mask is true. Without out, skipped values are undefined.
    casting : Casting, optional
        Casting rule for operands and output.
    order : Order, optional
        Output memory layout.
    dtype : None, optional
        Default Boolean dtype.
    subok : bool, optional
        Preserve NumPy array subclasses.

    Returns
    -------
    Tensor
        Boolean result.

    Raises
    ------
    DeviceMismatch
        A tensor and another operand use different devices when device is None.
    CuPyNotFound
        CUDA requested without CuPy.
    """
    return _compare(
        x1,
        x2,
        out=out,
        func_name="less_equal",
        device=device,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )
