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
    requires_grad: bool = False,
) -> Union[tuple[TensorType, ...], TensorType]:
    from ...tensor import Tensor

    if device is None:
        device = get_device(condition)

    # * One parameter overload
    if (x is None) and (y is None):
        data = to_tensordata(condition, device=device)
        if device == "cpu":
            result = np.where(data)  # type: ignore
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data)
        return tuple([Tensor(array, requires_grad=requires_grad) for array in result])
    # * Two parameters overload
    elif x is not None and y is not None:
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
        return Tensor(result, requires_grad=requires_grad)
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

    op_device = get_two_operand_op_device(x1, x2, device)

    x1, x2 = to_tensordata(x1, device=op_device), to_tensordata(x2, device=op_device)
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
    return Tensor(y)


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
