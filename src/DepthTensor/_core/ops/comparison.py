from typing import Union, Optional, Tuple, overload

from ...typing import (
    TensorLike,
    DeviceLike,
    NDArrayLikeBool,
    Casting,
    Order,
    OperandLike,
)

from ..exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

from ..utils import to_xp_array, get_device, get_two_operand_op_device

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
    condition: OperandLike,
    /,
    *,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> Tuple[TensorLike, ...]: ...


@overload
def where(
    condition: OperandLike,
    x: Optional[OperandLike],
    y: Optional[OperandLike],
    /,
    *,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> TensorLike: ...


def where(
    condition: OperandLike,
    x: Optional[OperandLike] = None,
    y: Optional[OperandLike] = None,
    /,
    *,
    device: Optional[DeviceLike] = None,
    requires_grad: bool = False,
) -> Union[Tuple[TensorLike, ...], TensorLike]:
    from ...tensor import Tensor

    if device is None:
        device = get_device(condition)

    # * One parameter overload
    if (x is None) and (y is None):
        data = to_xp_array(condition, device=device)
        if device == "cpu":
            result = np.where(data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data)
        return tuple(
            [
                Tensor(array, device=device, requires_grad=requires_grad)
                for array in result
            ]
        )
    # * Two parameters overload
    elif x is not None and y is not None:
        if (
            not (get_device(x) == get_device(y) == device)
            and not isinstance(x, (int, float, list, tuple))
            and not isinstance(y, (int, float, list, tuple))
        ):
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)

        data = to_xp_array(condition, device=device)
        x_data = to_xp_array(x, device=device)
        y_data = to_xp_array(y, device=device)
        if device == "cpu":
            result = np.where(data, x_data, y_data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data, x_data, y_data)
        return Tensor(result, device=device, requires_grad=requires_grad)
    else:
        raise ValueError("Both x and y parameters must be given.")


###
###
###


def wrapper_2in_1out(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    func_name: str,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
    from ...tensor import Tensor

    op_device = get_two_operand_op_device(x1, x2, device)

    x1, x2 = to_xp_array(x1, device=op_device), to_xp_array(x2, device=op_device)
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
    return Tensor(y, device=op_device)


def equal(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
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
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
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
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
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
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
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
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
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
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[NDArrayLikeBool] = None,
    *,
    device: Optional[DeviceLike] = None,
    where: Union[bool, NDArrayLikeBool] = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: None = None,
    subok: bool = True,
) -> TensorLike:
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
