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

from ..utils import to_tensordata, get_device, get_two_operand_op_device

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


@overload
def where(
    condition: TensorLike,
    /,
    *,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> Tuple[TensorType, ...]: ...


@overload
def where(
    condition: TensorLike,
    x: TensorLike | None,
    y: TensorLike | None,
    /,
    *,
    device: Device = "cpu",
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
    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition : TensorLike
        Where True, yield `x`, otherwise yield `y`.
    x : TensorLike | None, optional
        Values from which to choose.
    y : TensorLike | None, optional
        Values from which to choose.
    device : Device | None, optional
        The device to place the result on. If None, inferred from inputs.
    requires_grad : bool | None, optional
        Whether the result requires gradient computation. If None, inferred from inputs (x, y).

    Returns
    -------
    TensorType | tuple[TensorType, ...]
        A tensor with elements from `x` where `condition` is True, and elements
        from `y` elsewhere. If `x` and `y` are None, returns a tuple of
        indices where `condition` is True.
    """
    from ...tensor import Tensor

    if device is None:
        device = get_device(condition)

    # * One parameter overload
    if (x is None) and (y is None):
        if requires_grad is None:
            requires_grad = False
            
        data = to_tensordata(condition, device=device)
        if device == "cpu":
            result = np.where(data)  # type: ignore
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data)
        return tuple([Tensor._fast_init(array, device=device, requires_grad=requires_grad) for array in result])
    # * Two parameters overload
    elif x is not None and y is not None:
        if requires_grad is None:
            requires_grad = False
            if isinstance(x, Tensor):
                requires_grad = requires_grad or x.requires_grad
            if isinstance(y, Tensor):
                requires_grad = requires_grad or y.requires_grad

        if (
            not (get_device(x) == get_device(y) == device)
            and not isinstance(x, (int, float, list, tuple))
            and not isinstance(y, (int, float, list, tuple))
        ):
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)

        data = to_tensordata(condition, device=device)
        x_data = to_tensordata(x, device=device)
        y_data = to_tensordata(y, device=device)
        if device == "cpu":
            result = np.where(data, x_data, y_data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data, x_data, y_data)
        return Tensor._fast_init(result, device=device, requires_grad=requires_grad)
    else:
        raise ValueError("Both x and y parameters must be given.")


###
###
###


def wrapper_2in_1out(
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

    x1_is_tensor, x2_is_tensor = isinstance(x1, Tensor), isinstance(x2, Tensor)
    op_device = get_two_operand_op_device(x1, x2, x1_is_tensor, x2_is_tensor, device)

    x1 = to_tensordata(x1, device=op_device) if not x1_is_tensor else x1.data
    x2 = to_tensordata(x2, device=op_device) if not x2_is_tensor else x2.data

    if op_device == "cpu":
        y = getattr(np, func_name)(
            x1,
            x2,
            out=out,
            dtype=dtype,
            where=where,
            casting=casting,
            order=order,
            subok=subok,
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = getattr(cp, func_name)(x1, x2, out=out, dtype=dtype, casting=casting)
    return Tensor._fast_init(y, device=op_device)


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
    Return (x1 == x2) element-wise.

    Parameters
    ----------
    x1, x2 : TensorLike
        Input tensors.
    out : TensorDataBool | None, optional
        A location into which the result is stored.
    device : Device | None, optional
        The device to place the result on.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        Output array, element-wise comparison of x1 and x2.
    """
    return wrapper_2in_1out(
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
    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : TensorLike
        Input tensors.
    out : TensorDataBool | None, optional
        A location into which the result is stored.
    device : Device | None, optional
        The device to place the result on.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        Output array, element-wise comparison of x1 and x2.
    """
    return wrapper_2in_1out(
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
    Return the truth value of (x1 > x2) element-wise.

    Parameters
    ----------
    x1, x2 : TensorLike
        Input tensors.
    out : TensorDataBool | None, optional
        A location into which the result is stored.
    device : Device | None, optional
        The device to place the result on.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        Output array, element-wise comparison of x1 and x2.
    """
    return wrapper_2in_1out(
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
    Return the truth value of (x1 >= x2) element-wise.

    Parameters
    ----------
    x1, x2 : TensorLike
        Input tensors.
    out : TensorDataBool | None, optional
        A location into which the result is stored.
    device : Device | None, optional
        The device to place the result on.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        Output array, element-wise comparison of x1 and x2.
    """
    return wrapper_2in_1out(
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
    Return the truth value of (x1 < x2) element-wise.

    Parameters
    ----------
    x1, x2 : TensorLike
        Input tensors.
    out : TensorDataBool | None, optional
        A location into which the result is stored.
    device : Device | None, optional
        The device to place the result on.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        Output array, element-wise comparison of x1 and x2.
    """
    return wrapper_2in_1out(
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
    Return the truth value of (x1 <= x2) element-wise.

    Parameters
    ----------
    x1, x2 : TensorLike
        Input tensors.
    out : TensorDataBool | None, optional
        A location into which the result is stored.
    device : Device | None, optional
        The device to place the result on.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        Output array, element-wise comparison of x1 and x2.
    """
    return wrapper_2in_1out(
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


###
###
###

__all__ = [
    "where",
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
]
