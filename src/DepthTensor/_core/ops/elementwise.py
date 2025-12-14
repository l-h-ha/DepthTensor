from typing import Any

from ...typing import (
    TensorType,
    DTypeLike,
    Casting,
    Order,
    Axis,
    TensorDataBool,
    TensorData,
    TensorLike,
    Device,
    BinaryOp,
    BinaryDiff,
    BinaryFunc,
    UnaryOp,
    UnaryDiff,
    UnaryFunc,
)

from ..exceptions import (
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
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


def get_requires_grad_and_prev(x1: TensorLike, x2: TensorLike):
    from ...tensor import Tensor

    y_requires_grad = False
    prev = ()
    if isinstance(x1, Tensor):
        y_requires_grad = x1.requires_grad
        prev = (x1,)
    if isinstance(x2, Tensor):
        y_requires_grad = y_requires_grad or x2.requires_grad
        if len(prev) == 1:
            prev = (x1, x2)
        else:
            prev = (x2,)
    return y_requires_grad, prev


def create_2in_1out(op: BinaryOp, diff: BinaryDiff | None) -> BinaryFunc:
    def func(
        x1: TensorLike,
        x2: TensorLike,
        *,
        device: Device | None = None,
        in_place: bool = False,
        requires_grad: bool = True,
        **kwds: Any,
    ) -> TensorType:
        from ...tensor import Tensor

        op_device = get_two_operand_op_device(x1, x2, device)
        a1, a2 = to_tensordata(x1, device=op_device), to_tensordata(
            x2, device=op_device
        )
        y = op(a1, a2, device=op_device, **kwds)

        if in_place and isinstance(x1, Tensor):
            if isinstance(y, Tensor):
                x1.data = y.data
                return x1
            else:
                #! y could be an ArrayLike, so we convert it to a TensorData
                x1.data = to_tensordata(y)
                return x1

        y_requires_grad, prev = get_requires_grad_and_prev(x1, x2)
        if isinstance(y, Tensor):
            y.to_device(op_device, in_place=True)
            y.prev = prev
            if y_requires_grad and requires_grad:
                y.make_differentiable()
        else:
            y = Tensor(
                y,
                prev=prev,
                requires_grad=y_requires_grad and requires_grad,
            )
        if diff is None:
            if y_requires_grad and requires_grad:
                raise RuntimeError(
                    "Y is differentiable, but no differentiation function is given."
                )
            return y
        else:
            if y.requires_grad:
                if y.grad is None:
                    y.zeros_grad()
                dx1, dx2 = diff(y, a1, a2, **kwds)

                def backward() -> None:
                    if isinstance(x1, Tensor) and x1.requires_grad:
                        if x1.grad is None:
                            x1.zeros_grad()
                        x1.grad += dx1()  # type: ignore (y.grad cannot be None)
                    if isinstance(x2, Tensor) and x2.requires_grad:
                        if x2.grad is None:
                            x2.zeros_grad()
                        x2.grad += dx2()  # type: ignore (y.grad cannot be None)

                y.backward = backward
            return y

    return func


def create_1in_1out(op: UnaryOp, diff: UnaryDiff | None) -> UnaryFunc:
    def func(
        x: TensorLike,
        *,
        device: Device | None = None,
        in_place: bool = False,
        requires_grad: bool = True,
        **kwds: Any,
    ) -> TensorType:
        from ...tensor import Tensor

        if device is None:
            device_op = get_device(x)
        else:
            device_op = device

        a = to_tensordata(x, device=device_op)
        y = op(a, device=device_op, **kwds)

        y_requires_grad = False
        if isinstance(x, Tensor):
            if in_place:
                #! y is a TensorLike, so we convert it to a TensorData
                x.data = to_tensordata(y)
                return x
            y_requires_grad = x.requires_grad

        if isinstance(y, Tensor):
            y.to_device(device_op)
            y.prev = (x,)
            if y_requires_grad and requires_grad:
                y.make_differentiable()
        else:
            y = Tensor(
                y,
                prev=(x,),
                requires_grad=y_requires_grad and requires_grad,
            )
        if diff is None:
            if y_requires_grad and requires_grad:
                raise RuntimeError(
                    "Y is differentiable, but no differentiation function is given."
                )
            return y
        else:
            if y.requires_grad:
                if y.grad is None:
                    y.zeros_grad()
                dx = diff(y, a, **kwds)

                def backward() -> None:
                    if isinstance(x, Tensor) and x.requires_grad:
                        if x.grad is None:
                            x.zeros_grad()
                        x.grad += dx()  # type: ignore (y.grad cannot be None)

                y.backward = backward
            return y

    return func


def wrapper_2in_1out(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    func_name: str,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    """Wrapper for two-inputs-one-output functions.

    If the device parameter is None, the result's device adheres to the operands. However, if it is None, both operands are converted to the given device, resulting in a tensor of given device.
    """

    from ...tensor import Tensor

    op_device = get_two_operand_op_device(x1, x2, device)
    a1, a2 = to_tensordata(x1, op_device), to_tensordata(x2, op_device)

    if op_device == "cpu":
        kwds = {
            "out": out,
            "dtype": dtype,
            "where": where,
            "casting": casting,
            "order": order,
            "subok": subok,
        }
        if func_name == "matmul":
            del kwds["where"]
        y = getattr(np, func_name)(a1, a2, **kwds)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = getattr(cp, func_name)(a1, a2, out=out, dtype=dtype, casting=casting)

    if in_place and isinstance(x1, Tensor):
        x1.data = y
        return x1

    requires_grad, prev = get_requires_grad_and_prev(x1, x2)
    return Tensor(y, prev=prev, requires_grad=requires_grad)


def wrapper_1in_1out(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    func_name: str,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(x)
    else:
        device_op = device

    a = to_tensordata(x, device=device_op)
    if device_op == "cpu":
        y = getattr(np, func_name)(
            a,
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
        y = getattr(cp, func_name)(a, out=out, dtype=dtype, casting=casting)

    requires_grad = False
    if isinstance(x, Tensor):
        if in_place:
            x.data = y
            return x
        requires_grad = x.requires_grad
    return Tensor(y, prev=(x,), requires_grad=requires_grad)


###
### Arithmetics
###


def add(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_2in_1out(
        x1,
        x2,
        out=out,
        func_name="add",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def subtract(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_2in_1out(
        x1,
        x2,
        out=out,
        func_name="subtract",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def multiply(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_2in_1out(
        x1,
        x2,
        out=out,
        func_name="multiply",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def matmul(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_2in_1out(
        x1,
        x2,
        out=out,
        func_name="matmul",
        device=device,
        in_place=in_place,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def divide(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_2in_1out(
        x1,
        x2,
        out=out,
        func_name="divide",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def power(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_2in_1out(
        x1,
        x2,
        out=out,
        func_name="power",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def negative(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="negative",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def sign(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="sign",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def abs(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="abs",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def clip(
    a: TensorLike,
    a_min: TensorLike,
    a_max: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    requires_grad: bool = False,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    from ...tensor import Tensor

    is_tensor_op = False
    if (
        isinstance(a, Tensor)
        and isinstance(a_min, Tensor)
        and isinstance(a_max, Tensor)
    ):
        if not (a.device == a_min.device == a_max.device):
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)
        is_tensor_op = True

    if is_tensor_op and isinstance(a, Tensor):
        device_op = a.device
    else:
        device_op = device

    arr_a, arr_min, arr_max = (
        to_tensordata(a, device=device_op),
        to_tensordata(a_min, device=device_op),
        to_tensordata(a_max, device=device_op),
    )
    if device_op == "cpu":
        if out is None:
            y = np.clip(
                arr_a,
                arr_min,
                arr_max,
                where=where,
                casting=casting,
                order=order,
                dtype=dtype,
                subok=subok,
            )
        else:
            y = np.clip(
                arr_a,
                arr_min,
                arr_max,
                out=out,
                where=where,
                casting=casting,
                order=order,
                dtype=dtype,
                subok=subok,
            )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.clip(arr_a, arr_min, arr_max, out=out)
    return Tensor(y, requires_grad=requires_grad)


###
### Exponents/Logarithms
###


def exp(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="exp",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def sqrt(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="sqrt",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def log(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="log",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def square(
    x: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    return wrapper_1in_1out(
        x,
        out=out,
        func_name="square",
        device=device,
        in_place=in_place,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


###
###
###


def mean(
    x: TensorLike,
    /,
    axis: Axis | None = None,
    dtype: DTypeLike | None = None,
    out: TensorData | None = None,
    keepdims: bool = False,
    *,
    device: Device | None = None,
    in_place: bool = False,
    where: TensorDataBool | bool = True,
) -> TensorType:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(x)
    else:
        device_op = device

    a = to_tensordata(x, device=device_op)
    if device_op == "cpu":
        y = np.mean(
            a,
            out=out,
            dtype=dtype,
            where=where,
            axis=axis,
            keepdims=keepdims,
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = cp.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    requires_grad = False
    if isinstance(x, Tensor):
        if in_place:
            x.data = y
            return x
        requires_grad = x.requires_grad
    return Tensor(y, prev=(x,), requires_grad=requires_grad)


###
###
###

__all__ = [
    "add",
    "subtract",
    "multiply",
    "matmul",
    "divide",
    "power",
    "negative",
    "sign",
    "abs",
    "exp",
    "sqrt",
    "log",
    "square",
    "mean",
    "clip",
    "create_2in_1out",
    "create_1in_1out",
]
