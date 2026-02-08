from typing import Callable

from ...typing import (
    TensorType,
    DTypeLike,
    Casting,
    Order,
    Axis,
    Shape,
    TensorDataBool,
    TensorData,
    TensorLike,
    Device,
)

from ..exceptions import (
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
)

from ..utils import (
    to_tensordata,
    get_device,
    get_two_operand_op_device,
    unbroadcast_tensordata_to_shape,
)
from .function import Function

import numpy as np
import math

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


def get_requires_grad_and_prev(
    x1: TensorLike, x2: TensorLike, x1_is_tensor: bool, x2_is_tensor: bool
):
    from ...tensor import Tensor

    y_requires_grad = False
    if x1_is_tensor:
        y_requires_grad = x1.requires_grad
    if x2_is_tensor:
        y_requires_grad = y_requires_grad or x2.requires_grad
    return y_requires_grad


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

    x1_is_tensor, x2_is_tensor = isinstance(x1, Tensor), isinstance(x2, Tensor)

    op_device = get_two_operand_op_device(x1, x2, x1_is_tensor, x2_is_tensor, device)
    a1 = to_tensordata(x1, op_device) if not x1_is_tensor else x1.data
    a2 = to_tensordata(x2, op_device) if not x2_is_tensor else x2.data

    if op_device == "cpu":
        kwargs = {
            "out": out,
            "dtype": dtype,
            "where": where,
            "casting": casting,
            "order": order,
            "subok": subok,
        }
        if func_name == "matmul":
            del kwargs["where"]
        y = getattr(np, func_name)(a1, a2, **kwargs)
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = getattr(cp, func_name)(a1, a2, out=out, dtype=dtype, casting=casting)

    if in_place and x1_is_tensor:
        if x1.requires_grad:
            raise RuntimeError(
                "In-place operations are forbidden on differentiable tensors."
            )
        x1.data = y
        return x1

    requires_grad = get_requires_grad_and_prev(x1, x2, x1_is_tensor, x2_is_tensor)
    return Tensor._fast_init(y, device=op_device, requires_grad=requires_grad)


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
            if x.requires_grad:
                raise RuntimeError(
                    "In-place operations are forbidden on differentiable tensors."
                )
            x.data = y
            return x
        requires_grad = x.requires_grad
    return Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)


def wrapper_diff_2in_1out(
    y: TensorType,
    x1: TensorLike,
    x2: TensorLike,
    callback_x1: Callable,
    callback_x2: Callable,
) -> None:
    if not y.requires_grad:
        return

    from ...tensor import Tensor

    x1_is_tensor, x2_is_tensor = isinstance(x1, Tensor), isinstance(x2, Tensor)

    def backward() -> None:
        if y.grad is None:
            y.zero_grad()

        y_grad: TensorData = y.grad  # type: ignore (y.grad cannot be None)
        device = get_two_operand_op_device(x1, x2, x1_is_tensor, x2_is_tensor, None)
        x1_data, x2_data = to_tensordata(x1, device=device), to_tensordata(
            x2, device=device
        )

        if x1_is_tensor and x1.requires_grad:
            if x1.grad is None:
                x1.zero_grad()
            x1.grad += callback_x1(y_grad, x1.shape, device, x1_data, x2_data)
        if x2_is_tensor and x2.requires_grad:
            if x2.grad is None:
                x2.zero_grad()
            x2.grad += callback_x2(y_grad, x2.shape, device, x1_data, x2_data)

    prev = []
    if x1_is_tensor and x1.requires_grad:
        prev.append(x1)
    if x2_is_tensor and x2.requires_grad:
        prev.append(x2)
    y.prev = tuple(prev)
    y.backward = backward


def wrapper_diff_1in_1out(y: TensorType, x1: TensorLike, callback_x1: Callable) -> None:
    if not y.requires_grad:
        return

    from ...tensor import Tensor

    def backward() -> None:
        if y.grad is None:
            y.zero_grad()

        result_grad: TensorData = y.grad  # type: ignore
        _x = to_tensordata(x1)
        if isinstance(x1, Tensor) and x1.requires_grad:
            if x1.grad is None:
                x1.zero_grad()
            x1.grad += callback_x1(result_grad, x1.shape, x1.device, _x)

    if isinstance(x1, Tensor) and x1.requires_grad:
        y.prev = (x1,)
    y.backward = backward


###
### Arithmetics
###


class add_cls(Function):
    def link(self, y: TensorType, x1: TensorLike, x2: TensorLike) -> None:
        def callback_x1(y_grad, x1_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(y_grad, x1_shape, device)

        def callback_x2(y_grad, x2_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(y_grad, x2_shape, device)

        wrapper_diff_2in_1out(y, x1, x2, callback_x1, callback_x2)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Add arguments element-wise.

        Parameters
        ----------
        x1, x2 : TensorLike
            The arrays to be added.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The sum of x1 and x2, element-wise.
        """
        y = wrapper_2in_1out(
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
        if differentiate:
            self.link(y, x1, x2)
        return y


add = add_cls()


class sub_cls(Function):
    def link(self, y: TensorType, x1: TensorLike, x2: TensorLike) -> None:
        def callback_x1(y_grad, x1_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(y_grad, x1_shape, device)

        def callback_x2(y_grad, x2_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(-y_grad, x2_shape, device)

        wrapper_diff_2in_1out(y, x1, x2, callback_x1, callback_x2)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Subtract arguments, element-wise.

        Parameters
        ----------
        x1, x2 : TensorLike
            The arrays to be subtracted.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The difference of x1 and x2, element-wise.
        """
        y = wrapper_2in_1out(
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
        if differentiate:
            self.link(y, x1, x2)
        return y


class mul_cls(Function):
    def link(self, y: TensorType, x1: TensorLike, x2: TensorLike) -> None:
        def callback_x1(y_grad, x1_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(y_grad * x2_data, x1_shape, device)

        def callback_x2(y_grad, x2_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(y_grad * x1_data, x2_shape, device)

        wrapper_diff_2in_1out(y, x1, x2, callback_x1, callback_x2)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Multiply arguments element-wise.

        Parameters
        ----------
        x1, x2 : TensorLike
            Input arrays to be multiplied.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The product of x1 and x2, element-wise.
        """
        y = wrapper_2in_1out(
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
        if differentiate:
            self.link(y, x1, x2)
        return y


class matmul_cls(Function):
    def link(self, y: TensorType, x1: TensorLike, x2: TensorLike) -> None:
        def callback_x1(y_grad, x1_shape, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            grad = xp.matmul(y_grad, xp.swapaxes(x2_data, -1, -2))
            return unbroadcast_tensordata_to_shape(grad, x1_shape, device)

        def callback_x2(y_grad, x2_shape, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            grad = xp.matmul(xp.swapaxes(x1_data, -1, -2), y_grad)
            return unbroadcast_tensordata_to_shape(grad, x2_shape, device)

        wrapper_diff_2in_1out(y, x1, x2, callback_x1, callback_x2)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Matrix product of two arrays.

        Parameters
        ----------
        x1, x2 : TensorLike
            Input arrays.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The matrix product of the inputs.
        """
        # matmul does not support 'where'
        y = wrapper_2in_1out(
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
        if differentiate:
            self.link(y, x1, x2)
        return y


class div_cls(Function):
    def link(self, y: TensorType, x1: TensorLike, x2: TensorLike) -> None:
        def callback_x1(y_grad, x1_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(y_grad / x2_data, x1_shape, device)

        def callback_x2(y_grad, x2_shape, device, x1_data, x2_data) -> TensorData:
            return unbroadcast_tensordata_to_shape(
                -y_grad * x1_data / (x2_data**2), x2_shape, device
            )

        wrapper_diff_2in_1out(y, x1, x2, callback_x1, callback_x2)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Returns a true division of the inputs, element-wise.

        Parameters
        ----------
        x1 : TensorLike
            Dividend array.
        x2 : TensorLike
            Divisor array.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            This is a scalar if both x1 and x2 are scalars.
        """
        y = wrapper_2in_1out(
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
        if differentiate:
            self.link(y, x1, x2)
        return y


class power_cls(Function):
    def link(self, y: TensorType, x1: TensorLike, x2: TensorLike) -> None:
        def callback_x1(y_grad, x1_shape, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            grad = y_grad * x2_data * xp.power(x1_data, x2_data - 1)
            return unbroadcast_tensordata_to_shape(grad, x1_shape, device)

        def callback_x2(y_grad, x2_shape, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            grad = y_grad * xp.power(x1_data, x2_data) * xp.log(x1_data)
            return unbroadcast_tensordata_to_shape(grad, x2_shape, device)

        wrapper_diff_2in_1out(y, x1, x2, callback_x1, callback_x2)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        First array elements raised to powers from second array, element-wise.

        Parameters
        ----------
        x1 : TensorLike
            The bases.
        x2 : TensorLike
            The exponents.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The bases in x1 raised to the exponents in x2.
        """
        y = wrapper_2in_1out(
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
        if differentiate:
            self.link(y, x1, x2)
        return y


###
### Unary Ops
###


class negative_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            return -y_grad

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Numerical negative, element-wise.

        Parameters
        ----------
        x : TensorLike
            Input array.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            Returned array or scalar: y = -x.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


class sign_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return xp.zeros_like(x_data)

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Returns an element-wise indication of the sign of a number.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The sign of x.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


class abs_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return y_grad * xp.sign(x_data)

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Calculate the absolute value element-wise.

        Parameters
        ----------
        x : TensorLike
            Input array.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            An array containing the absolute value of each element in x.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


class exp_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return y_grad * xp.exp(x_data)

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Calculate the exponential of all elements in the input array.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            Output array, element-wise exponential of x.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


class sqrt_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return y_grad / (2 * xp.sqrt(x_data))

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Return the non-negative square-root of an array, element-wise.

        Parameters
        ----------
        x : TensorLike
            The values whose square-roots are required.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            An array of the same shape as x, containing the positive square-root
            of each element in x.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


class log_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            return y_grad / x_data

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Natural logarithm, element-wise.

        Parameters
        ----------
        x : TensorLike
            Input value.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The natural logarithm of x, element-wise.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


class square_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            return y_grad * 2 * x_data

        wrapper_diff_1in_1out(y, x, callback)

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Return the element-wise square of the input.

        Parameters
        ----------
        x : TensorLike
            Input data.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            Element-wise x*x, of the same shape and dtype as x.
        """
        y = wrapper_1in_1out(
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
        if differentiate:
            self.link(y, x)
        return y


###
### Special
###


def clip(
    a: TensorLike,
    a_min: TensorLike,
    a_max: TensorLike,
    /,
    out: TensorData | None = None,
    *,
    requires_grad: bool | None = None,
    device: Device | None = None,
    where: TensorDataBool | bool = True,
    casting: Casting = "same_kind",
    order: Order = "K",
    dtype: DTypeLike | None = None,
    subok: bool = True,
) -> TensorType:
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges.
    For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    Parameters
    ----------
    a : TensorLike
        Array containing elements to clip.
    a_min : TensorLike
        Minimum value. If None, clipping is not performed on lower interval edge.
        Not more than one of `a_min` and `a_max` may be None.
    a_max : TensorLike
        Maximum value. If None, clipping is not performed on upper interval edge.
        Not more than one of `a_min` and `a_max` may be None.
    out : TensorData | None, optional
        The results will be placed in this array.
    requires_grad : bool | None, optional
        Whether the result requires gradient computation. If None, inferred from inputs.
    device : Device | None, optional
        The device to place the result on. If None, inferred from inputs.
    where : TensorDataBool | bool, optional
        This condition is broadcast over the input.
    casting : Casting, optional
        Controls what kind of data casting may occur.
    order : Order, optional
        Controls the memory layout of the result.
    dtype : DTypeLike | None, optional
        Overrides the data type of the result.
    subok : bool, optional
        If True, then sub-classes will be passed-through.

    Returns
    -------
    TensorType
        An array with the elements of `a`, but where values < `a_min` are replaced with `a_min`,
        and those > `a_max` with `a_max`.
    """
    from ...tensor import Tensor

    if (
        isinstance(a, Tensor)
        and isinstance(a_min, Tensor)
        and isinstance(a_max, Tensor)
    ):
        if not (a.device == a_min.device == a_max.device):
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device

    if requires_grad is None:
        requires_grad = False
        if isinstance(a, Tensor):
            requires_grad = requires_grad or a.requires_grad
        if isinstance(a_min, Tensor):
            requires_grad = requires_grad or a_min.requires_grad
        if isinstance(a_max, Tensor):
            requires_grad = requires_grad or a_max.requires_grad

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
    return Tensor._fast_init(y, device=device_op, requires_grad=requires_grad)


class mean_cls(Function):
    def link(
        self, y: TensorType, x: TensorType, axis: Axis | None, keepdims: bool
    ) -> None:
        def callback_x1(
            y_grad: TensorData, x1_shape: Shape, device: Device
        ) -> TensorData:
            if device == "cpu":
                xp = np
            else:
                if cp is None:
                    raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                xp = cp

            inp_size = math.prod(x1_shape)
            grad_size = y_grad.size
            N = inp_size / grad_size
            grad = y_grad * (1.0 / N)

            if not keepdims and axis is not None:
                grad = xp.expand_dims(grad, axis)
            return xp.broadcast_to(grad, x1_shape)

        def backward() -> None:
            if y.requires_grad:
                if y.grad is None:
                    y.zero_grad()
                if x.grad is None:
                    x.zero_grad()
                x.grad += callback_x1(y.grad, x.shape, y.device)

        from ...tensor import Tensor

        if isinstance(x, Tensor) and x.requires_grad:
            y.prev = (x,)
        y.backward = backward

    def __call__(
        self,
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
        differentiate: bool = False,
    ) -> TensorType:
        """
        Compute the arithmetic mean along the specified axis.

        Parameters
        ----------
        x : TensorLike
            Array containing numbers whose mean is desired.
        axis : Axis | None, optional
            Axis or axes along which the means are computed.
        dtype : DTypeLike | None, optional
            Type to use in computing the mean.
        out : TensorData | None, optional
            Alternate output array in which to place the result.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            Elements to include in the mean.
        differentiate : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            A new array containing the mean values.
        """
        op_device = device
        if device is None:
            op_device = get_device(x)

        a = to_tensordata(x, device=op_device)
        if op_device == "cpu":
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

        from ...tensor import Tensor

        requires_grad = False
        if isinstance(x, Tensor):
            if in_place:
                x.data = y
                return x
            requires_grad = x.requires_grad

            y = Tensor._fast_init(y, device=op_device, requires_grad=requires_grad)
            if differentiate:
                self.link(y, x, axis, keepdims)
        else:
            y = Tensor._fast_init(y, device=op_device, requires_grad=requires_grad)
        return y


add = add_cls()
subtract = sub_cls()
multiply = mul_cls()
matmul = matmul_cls()
divide = div_cls()
power = power_cls()
negative = negative_cls()
sign = sign_cls()
abs = abs_cls()
exp = exp_cls()
sqrt = sqrt_cls()
log = log_cls()
square = square_cls()
mean = mean_cls()


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
]
