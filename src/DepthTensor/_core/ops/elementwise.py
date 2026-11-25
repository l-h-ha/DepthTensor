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
    _accumulate_operand_grad,
    _cast_gradient_work,
    _cast_compute_operand,
    _check_dtype,
    _pointwise_mask,
    _cast_backward_inputs,
    _save_operand_data,
    to_tensordata,
    get_device,
    get_two_operand_op_device,
)
from ..output import (
    _cuda_pointwise,
    _finish_output,
    _numpy_pointwise,
    _temporary_out,
    _prepare_output,
    _write_output,
)
from .function import Function
from .reduction import _link_reduction_backward, _prepare_reduction, _reduced_shape

import numpy as np
import math

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def _apply_arithmetic(
    operands: tuple[TensorLike, ...],
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
    link_backward: Callable | None = None,
    requires_grad: bool | None = None,
) -> TensorType:
    from ...tensor import Tensor, _check_grad_dtype

    _check_dtype(dtype)

    tensor_inputs = tuple(isinstance(operand, Tensor) for operand in operands)
    tracks_inputs = any(
        is_tensor and operand.requires_grad for operand, is_tensor in zip(operands, tensor_inputs)
    )
    if requires_grad is None:
        requires_grad = link_backward is not None and tracks_inputs

    destination = operands[0] if in_place and tensor_inputs[0] else None
    if destination is not None and (destination.requires_grad or requires_grad):
        raise RuntimeError("In-place operations are forbidden on differentiable tensors.")

    if requires_grad and dtype is not None:
        _check_grad_dtype(np.dtype(dtype))

    if device is not None:
        op_device = device
    elif len(operands) == 1:
        op_device = get_device(operands[0])
    else:
        op_device = get_two_operand_op_device(*operands, *tensor_inputs, device=None)

    out, out_aliases = _prepare_output(out, op_device, *operands, where, requires_grad=requires_grad)

    values = tuple(to_tensordata(operand, op_device) for operand in operands)
    mask = _pointwise_mask(where, op_device, save=requires_grad and tracks_inputs)
    supports_where = func_name != "matmul"

    if op_device == "cpu":
        y = _numpy_pointwise(
            getattr(np, func_name), values, out, mask, supports_where=supports_where,
            order=order, dtype=dtype, casting=casting, subok=subok,
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

        y = _cuda_pointwise(
            getattr(cp, func_name), values, out, mask, supports_where=supports_where,
            order=order, dtype=dtype, casting=casting,
        )

    result = _finish_output(y, op_device, requires_grad, out_aliases, destination=destination)

    if requires_grad and tracks_inputs and link_backward is not None:
        link_backward(result, *operands, mask, dtype=dtype)

    return result

def _link_arithmetic_backward(
    y: TensorType,
    operands: tuple[TensorLike, ...],
    derivatives: tuple[Callable, ...],
    mask: TensorDataBool | None = None,
    dtype: DTypeLike | None = None,
    *,
    save_inputs: bool = True,
    mask_zero_seed: bool = False,
) -> None:
    if not y.requires_grad:
        return

    from ...tensor import Tensor

    versions = tuple((x, x._version) for x in operands if isinstance(x, Tensor))
    tracked = tuple(
        (x, derivative) for x, derivative in zip(operands, derivatives)
        if isinstance(x, Tensor) and x.requires_grad
    )

    grad_dtypes = [x.dtype for x, _ in tracked]
    if len(tracked) < len(operands):
        grad_dtypes.append(y.dtype)
    grad_dtype = np.result_type(*grad_dtypes)
    device = y.device

    if not save_inputs:
        saved_values = (None,) * len(operands)
    elif len(operands) == 2 and operands[0] is operands[1]:
        saved_values = (_save_operand_data(operands[0], device, dtype),) * 2
    else:
        saved_values = tuple(_save_operand_data(x, device, dtype) for x in operands)

    def grad_fn() -> None:
        for x, version in versions:
            if x._version != version:
                raise RuntimeError("A saved operand was modified by an in-place operation.")

        if y.grad is None:
            y.zero_grad()

        grad, values = _cast_backward_inputs(y.grad, grad_dtype, *saved_values)
        xp = np if device == "cpu" else cp

        if mask is not None:
            grad = xp.where(mask, grad, grad.dtype.type(0))

        # Zero seeds skip singular derivatives.
        active = grad != 0 if mask_zero_seed else mask
        if save_inputs and active is not None:
            values = tuple(xp.where(active, value, value.dtype.type(1)) for value in values)

        for x, derivative in tracked:
            if x.requires_grad:
                _accumulate_operand_grad(x, derivative(grad, device, *values))

    if tracked or len(operands) == 2:
        y.prev = tuple(x for x, _ in tracked)
    y.grad_fn = grad_fn

class add_cls(Function):
    def link(
        self, y: TensorType, x1: TensorLike, x2: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x1_data, x2_data) -> TensorData:
            return y_grad

        _link_arithmetic_backward(y, (x1, x2), (callback, callback), mask, dtype, save_inputs=False)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Add operands elementwise.

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
        in_place : bool, optional
            Replace x1's data if it is a Tensor.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        DeviceMismatch
            A tensor and another operand use different devices when device is None.
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x1, x2),
            out=out,
            func_name="add",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

add = add_cls()

class sub_cls(Function):
    def link(
        self, y: TensorType, x1: TensorLike, x2: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback_x1(y_grad, device, x1_data, x2_data) -> TensorData:
            return y_grad

        def callback_x2(y_grad, device, x1_data, x2_data) -> TensorData:
            return -y_grad

        _link_arithmetic_backward(
            y, (x1, x2), (callback_x1, callback_x2), mask, dtype, save_inputs=False
        )

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Subtract the second operand elementwise.

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
        in_place : bool, optional
            Replace x1's data if it is a Tensor.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        DeviceMismatch
            A tensor and another operand use different devices when device is None.
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x1, x2),
            out=out,
            func_name="subtract",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class mul_cls(Function):
    def link(
        self, y: TensorType, x1: TensorLike, x2: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback_x1(y_grad, device, x1_data, x2_data) -> TensorData:
            return y_grad * x2_data

        def callback_x2(y_grad, device, x1_data, x2_data) -> TensorData:
            return y_grad * x1_data

        _link_arithmetic_backward(
            y, (x1, x2), (callback_x1, callback_x2), mask, dtype, mask_zero_seed=True
        )

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Multiply operands elementwise.

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
        in_place : bool, optional
            Replace x1's data if it is a Tensor.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        DeviceMismatch
            A tensor and another operand use different devices when device is None.
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x1, x2),
            out=out,
            func_name="multiply",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class matmul_cls(Function):
    def link(
        self, y: TensorType, x1: TensorLike, x2: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def expand_gradient(grad, xp, left_is_vector, right_is_vector):
            # Restore vector axes dropped by matmul.
            if right_is_vector:
                grad = xp.expand_dims(grad, -1)

            if left_is_vector:
                grad = xp.expand_dims(grad, -2)

            return grad

        def multiply_gradient(grad, matrix, xp):
            if xp.all(xp.isfinite(matrix)) or xp.all(grad != 0):
                return xp.matmul(grad, matrix)

            # Skip matrix entries where each gradient row is zero.
            batch_shape = np.broadcast_shapes(grad.shape[:-2], matrix.shape[:-2])
            shape = (*batch_shape, grad.shape[-2], matrix.shape[-1])
            result = xp.empty(shape, dtype=xp.result_type(grad, matrix))

            for row in range(grad.shape[-2]):
                seed = grad[..., row:row + 1, :]
                active = xp.swapaxes(seed, -1, -2) != 0
                operand = xp.where(active, matrix, matrix.dtype.type(0))
                result[..., row:row + 1, :] = xp.matmul(seed, operand)

            return result

        def callback_x1(y_grad, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            left_is_vector = x1_data.ndim == 1
            right_is_vector = x2_data.ndim == 1

            grad = expand_gradient(y_grad, xp, left_is_vector, right_is_vector)
            right_matrix = x2_data[:, None] if right_is_vector else x2_data
            grad = multiply_gradient(grad, xp.swapaxes(right_matrix, -1, -2), xp)

            if left_is_vector:
                grad = xp.squeeze(grad, axis=-2)

            return grad

        def callback_x2(y_grad, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            left_is_vector = x1_data.ndim == 1
            right_is_vector = x2_data.ndim == 1

            grad = expand_gradient(y_grad, xp, left_is_vector, right_is_vector)
            left_matrix = x1_data[None, :] if left_is_vector else x1_data
            grad = multiply_gradient(xp.swapaxes(grad, -1, -2), left_matrix, xp)
            grad = xp.swapaxes(grad, -1, -2)

            if right_is_vector:
                grad = xp.squeeze(grad, axis=-1)

            return grad

        _link_arithmetic_backward(y, (x1, x2), (callback_x1, callback_x2), mask, dtype)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Multiply matrices, batches of matrices, or vectors.

        Parameters
        ----------
        x1 : TensorLike
            Left matrix, batch, or vector.
        x2 : TensorLike
            Right matrix, batch, or vector with a matching inner dimension.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace x1's data if it is a Tensor.
        casting : Casting, optional
            Casting rule for operands and output.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike, optional
            Operation dtype.
        subok : bool, optional
            Preserve NumPy array subclasses.
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        DeviceMismatch
            A tensor and another operand use different devices when device is None.
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x1, x2),
            out=out,
            func_name="matmul",
            device=device,
            in_place=in_place,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class div_cls(Function):
    def link(
        self, y: TensorType, x1: TensorLike, x2: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback_x1(y_grad, device, x1_data, x2_data) -> TensorData:
            return y_grad / x2_data

        def callback_x2(y_grad, device, x1_data, x2_data) -> TensorData:
            xp = np if device == "cpu" else cp
            # Combine mantissas before restoring exponents to avoid overflow.
            seed_fraction, seed_exponent = xp.frexp(y_grad)
            numerator_fraction, numerator_exponent = xp.frexp(x1_data)
            denominator_fraction, denominator_exponent = xp.frexp(x2_data)

            fraction = -(seed_fraction * numerator_fraction) / (
                denominator_fraction * denominator_fraction
            )
            exponent = seed_exponent + numerator_exponent - 2 * denominator_exponent
            grad = xp.ldexp(fraction, exponent)
            return grad

        _link_arithmetic_backward(
            y, (x1, x2), (callback_x1, callback_x2), mask, dtype, mask_zero_seed=True
        )

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Divide x1 by x2 elementwise.

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
        in_place : bool, optional
            Replace x1's data if it is a Tensor.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        DeviceMismatch
            A tensor and another operand use different devices when device is None.
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x1, x2),
            out=out,
            func_name="divide",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class power_cls(Function):
    def link(
        self, y: TensorType, x1: TensorLike, x2: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback_x1(y_grad, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            nonzero_exponent = x2_data != 0
            # Zero exponents give constants, including 0**0.
            safe_exponent = xp.where(nonzero_exponent, x2_data, x2_data.dtype.type(1))

            if x1_data.dtype.kind == "f":
                # Use p*x**p/x when x**(p - 1) may overflow.
                tiny = np.finfo(x1_data.dtype).tiny
                scale_by_base = (x1_data > 0) & (x1_data < tiny) & (safe_exponent < 0.5)
                if safe_exponent.dtype.kind == "i":
                    scale_by_base = scale_by_base | (
                        safe_exponent == np.iinfo(safe_exponent.dtype).min
                    )
                exponent_step_type = np.int8 if safe_exponent.dtype.kind == "b" else safe_exponent.dtype.type
                power_exponent = safe_exponent - xp.where(scale_by_base, exponent_step_type(0), exponent_step_type(1))
                divisor = xp.where(scale_by_base, x1_data, x1_data.dtype.type(1))
                derivative = (x2_data * xp.power(x1_data, power_exponent)) / divisor
            else:
                derivative = x2_data * xp.power(x1_data, safe_exponent - 1)

            y_grad = xp.where(nonzero_exponent, y_grad, y_grad.dtype.type(0))
            return y_grad * derivative

        def callback_x2(y_grad, device, x1_data, x2_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            # Zero base gives constant zero for positive exponents.
            zero_base = (x1_data == 0) & (x2_data > 0)
            safe_base = xp.where(zero_base, x1_data.dtype.type(1), x1_data)
            y_grad = xp.where(zero_base, y_grad.dtype.type(0), y_grad)

            log_base = xp.log(_cast_gradient_work(safe_base, y_grad.dtype))
            derivative = xp.power(safe_base, x2_data) * log_base
            return y_grad * derivative

        _link_arithmetic_backward(
            y, (x1, x2), (callback_x1, callback_x2), mask, dtype, mask_zero_seed=True
        )

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Raise each base to its corresponding exponent.

        Parameters
        ----------
        x1 : TensorLike
            Broadcastable bases.
        x2 : TensorLike
            Broadcastable exponents.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace x1's data if it is a Tensor.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        DeviceMismatch
            A tensor and another operand use different devices when device is None.
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x1, x2),
            out=out,
            func_name="power",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class negative_cls(Function):
    def link(
        self, y: TensorType, x: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            return -y_grad

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Negate each element.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="negative",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class sign_cls(Function):
    def link(
        self,
        y: TensorType,
        x: TensorLike,
        mask: TensorDataBool | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            xp = np if device == "cpu" else cp
            return xp.zeros_like(x_data)

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype, mask_zero_seed=True)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Return elementwise signs.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="sign",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class abs_cls(Function):
    def link(
        self, y: TensorType, x: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return y_grad * xp.sign(x_data)

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype, mask_zero_seed=True)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Return elementwise magnitudes.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
        where : TensorDataBool or bool, optional
            Write where the mask is true. Without out, skipped values are undefined.
        casting : Casting, optional
            Casting rule for operands and output.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike, optional
            Magnitude dtype.
        subok : bool, optional
            Preserve NumPy array subclasses.
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="abs",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class exp_cls(Function):
    def link(
        self, y: TensorType, x: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return y_grad * xp.exp(x_data)

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype, mask_zero_seed=True)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Compute the elementwise exponential.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="exp",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class sqrt_cls(Function):
    def link(
        self, y: TensorType, x: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            xp = cp if device != "cpu" and cp else np
            return y_grad / (2 * xp.sqrt(x_data))

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype, mask_zero_seed=True)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Compute the elementwise principal square root.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="sqrt",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class log_cls(Function):
    def link(
        self, y: TensorType, x: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            return y_grad / x_data

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype, mask_zero_seed=True)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Compute the elementwise natural logarithm.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="log",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class square_cls(Function):
    def link(
        self, y: TensorType, x: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        def callback(y_grad, device, x_data) -> TensorData:
            return y_grad * 2 * x_data

        _link_arithmetic_backward(y, (x,), (callback,), mask, dtype, mask_zero_seed=True)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Square each element.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Buffer on the operation's device.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
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
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Computed values.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        return _apply_arithmetic(
            (x,),
            out=out,
            func_name="square",
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            link_backward=self.link,
            requires_grad=requires_grad,
        )

class clip_cls(Function):
    def link(
        self, y: TensorType, a: TensorLike, a_min: TensorLike, a_max: TensorLike,
        mask: TensorDataBool | None = None, *, dtype: DTypeLike | None = None,
    ) -> None:
        from ...tensor import Tensor

        tracks_a = isinstance(a, Tensor) and a.requires_grad
        tracks_min = isinstance(a_min, Tensor) and a_min.requires_grad
        tracks_max = isinstance(a_max, Tensor) and a_max.requires_grad

        operand_versions = tuple(
            (operand, operand._version)
            for operand in (a, a_min, a_max)
            if isinstance(operand, Tensor)
        )

        device = y.device
        a_data = _save_operand_data(a, device, dtype)
        min_data = _save_operand_data(a_min, device, dtype) if a_min is not None else None
        max_data = _save_operand_data(a_max, device, dtype) if a_max is not None else None

        def grad_fn() -> None:
            if any(operand._version != version for operand, version in operand_versions):
                raise RuntimeError(
                    "A saved operand was modified by an in-place operation."
                )

            if y.grad is None:
                y.zero_grad()

            y_grad = y.grad
            xp = cp if device != "cpu" and cp else np
            zero = y_grad.dtype.type(0)

            if mask is not None:
                y_grad = xp.where(mask, y_grad, zero)

            # Equal or reversed bounds select the upper bound.
            mask_min = (
                a_data < min_data if min_data is not None else xp.zeros_like(a_data, dtype=bool)
            )
            mask_max = (
                a_data > max_data if max_data is not None else xp.zeros_like(a_data, dtype=bool)
            )

            if min_data is not None and max_data is not None:
                reversed_or_equal_bounds = min_data >= max_data
                mask_min = mask_min & ~reversed_or_equal_bounds
                mask_max = mask_max | reversed_or_equal_bounds

            mask_a = ~(mask_min | mask_max)

            if tracks_a and a.requires_grad:
                _accumulate_operand_grad(a, xp.where(mask_a, y_grad, zero))

            if tracks_min and a_min.requires_grad:
                _accumulate_operand_grad(a_min, xp.where(mask_min, y_grad, zero))

            if tracks_max and a_max.requires_grad:
                _accumulate_operand_grad(a_max, xp.where(mask_max, y_grad, zero))

        prev = []
        if tracks_a:
            prev.append(a)
        if tracks_min:
            prev.append(a_min)
        if tracks_max:
            prev.append(a_max)

        y.prev = tuple(prev)
        y.grad_fn = grad_fn

    def __call__(
        self,
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
        Clip values. Equal or reversed bounds select the upper bound.

        Parameters
        ----------
        a : TensorLike
            Input values.
        a_min : TensorLike or None
            Lower bound. None leaves it open.
        a_max : TensorLike or None
            Upper bound. None leaves it open if a_min is set.
        out : TensorData, optional
            Buffer on the operation's device.
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.
        device : {'cpu', 'cuda'}, optional
            If None, use the input device.
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
            Tensor input and Tensor bound use different devices.
        RuntimeError
            Gradients require a floating-point dtype. `out` cannot overlap a
            tracked tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        from ...tensor import Tensor, _check_grad_dtype

        _check_dtype(dtype)

        if isinstance(a, Tensor):
            if isinstance(a_min, Tensor) and a.device != a_min.device:
                raise DeviceMismatch(DEVICE_MISMATCH_MSG)

            if isinstance(a_max, Tensor) and a.device != a_max.device:
                raise DeviceMismatch(DEVICE_MISMATCH_MSG)

        if device is None:
            op_device = get_device(a)
        else:
            op_device = device

        if requires_grad is None:
            requires_grad = False
            if isinstance(a, Tensor):
                requires_grad = requires_grad or a.requires_grad
            if isinstance(a_min, Tensor):
                requires_grad = requires_grad or a_min.requires_grad
            if isinstance(a_max, Tensor):
                requires_grad = requires_grad or a_max.requires_grad

        if requires_grad and dtype is not None:
            _check_grad_dtype(np.dtype(dtype))

        arr_a = to_tensordata(a, device=op_device)
        arr_min = to_tensordata(a_min, device=op_device) if a_min is not None else None
        arr_max = to_tensordata(a_max, device=op_device) if a_max is not None else None
        mask = _pointwise_mask(where, op_device, save=requires_grad)

        out, out_aliases = _prepare_output(
            out, op_device, a, a_min, a_max, where, requires_grad=requires_grad
        )
        operands = (arr_a, arr_min, arr_max)

        if op_device == "cpu":
            y = _numpy_pointwise(
                np.clip, operands, out, mask, order=order, dtype=dtype, casting=casting, subok=subok
            )
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            # Check the requested casts before calling CuPy clip.
            operand_dtypes = tuple(value.dtype for value in operands if value is not None)
            compute_dtype = np.result_type(*operand_dtypes) if dtype is None else np.dtype(dtype)

            for operand_dtype in operand_dtypes:
                if not np.can_cast(operand_dtype, compute_dtype, casting=casting):
                    raise TypeError(
                        f"Cannot cast clip input {operand_dtype} to {compute_dtype} with casting={casting!r}."
                    )

            if dtype is not None:
                operands = tuple(
                    None if value is None else _cast_compute_operand(value, compute_dtype)
                    for value in operands
                )

            y = _cuda_pointwise(
                cp.clip, operands, out, mask, order=order,
                work_dtype=compute_dtype, output_casting=casting,
            )

        y_tensor = _finish_output(y, op_device, requires_grad, out_aliases)

        if requires_grad:
            self.link(y_tensor, a, a_min, a_max, mask, dtype=dtype)

        return y_tensor

def _mean_count(shape: Shape, axis: Axis | None) -> int:
    if axis is None:
        return math.prod(shape)

    axes = axis if isinstance(axis, tuple) else (axis,)
    return math.prod(shape[dim] for dim in axes)

class mean_cls(Function):
    def link(
        self,
        y: TensorType,
        x: TensorType,
        axis: Axis | None,
        keepdims: bool,
        mask: TensorDataBool | None = None,
        count: TensorData | None = None,
    ) -> None:
        if count is None:
            divisor = _mean_count(x.shape, axis) or 1
        else:
            xp = np if y.device == "cpu" else cp
            divisor = xp.maximum(count, 1)

        _link_reduction_backward(y, x, axis, keepdims, mask, divisor)

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
        requires_grad: bool | None = None,
    ) -> TensorType:
        """
        Average selected elements along one or more axes.

        Parameters
        ----------
        x : TensorLike
            Input values.
        axis : int or tuple of int, optional
            None reduces all axes.
        dtype : DTypeLike, optional
            Accumulator dtype.
        out : TensorData, optional
            Buffer on the operation's device.
        keepdims : bool, optional
            Keep reduced axes with size one.
        device : {'cpu', 'cuda'}, optional
            If None, choose the device from the inputs.
        in_place : bool, optional
            Replace the Tensor's data.
        where : TensorDataBool or bool, optional
            Average values where the mask is true.
        requires_grad : bool or None, optional
            If None, track gradients when an input does. False disables tracking.

        Returns
        -------
        Tensor
            Average for each reduced slice.

        Raises
        ------
        RuntimeError
            Gradients require a floating-point dtype. In-place writes disallow
            tracked destinations and results. `out` cannot overlap a tracked
            tensor, or any input if an input tracks gradients.
        CuPyNotFound
            CUDA requested without CuPy.
        """
        from ...tensor import Tensor, _check_grad_dtype

        _check_dtype(dtype)

        x_is_tensor = isinstance(x, Tensor)
        if requires_grad is None:
            requires_grad = x_is_tensor and x.requires_grad

        if in_place and x_is_tensor and (x.requires_grad or requires_grad):
            raise RuntimeError("In-place operations are forbidden on differentiable tensors.")

        if requires_grad and dtype is not None:
            _check_grad_dtype(np.dtype(dtype))

        op_device = get_device(x) if device is None else device
        a = to_tensordata(x, device=op_device)
        axis, keepdims, mask = _prepare_reduction(
            a, axis, keepdims, where, operation="mean", save_mask=requires_grad
        )

        out, aliases = _prepare_output(
            out, op_device, x, where, requires_grad=requires_grad,
            shape=_reduced_shape(a.shape, axis, keepdims) if out is not None else None,
        )

        temp_out = None
        if out is not None:
            # Divide before narrowing to out.dtype.
            accumulator = a.dtype if dtype is None else np.dtype(dtype)
            if dtype is None and a.dtype.kind in "biu":
                accumulator = np.dtype(np.float64)
            elif dtype is None and a.dtype == np.float16:
                accumulator = np.dtype(np.float32)

            work_dtype = np.result_type(accumulator, out.dtype)
            temp_out = _temporary_out(out, work_dtype)

        count = None
        if mask is not None and (requires_grad or op_device != "cpu"):
            count = mask.sum(axis=axis, keepdims=keepdims)

        if op_device == "cpu":
            y = np.mean(
                a,
                out=temp_out,
                dtype=dtype,
                where=True if mask is None else mask,
                axis=axis,
                keepdims=keepdims,
            )
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            explicit_half = dtype is not None and np.dtype(dtype) == np.float16
            if mask is None and not explicit_half:
                y = cp.mean(a, axis=axis, dtype=dtype, out=temp_out, keepdims=keepdims)
            else:
                acc_dtype = dtype
                result_dtype = a.dtype if dtype is None else dtype
                if dtype is None:
                    if a.dtype.kind in "biu":
                        acc_dtype = result_dtype = np.float64
                    elif a.dtype == np.float16:
                        acc_dtype = np.float32
                    elif temp_out is not None:
                        acc_dtype = temp_out.dtype

                values = a if mask is None else cp.where(mask, a, a.dtype.type(0))
                total = cp.sum(values, axis=axis, dtype=acc_dtype, out=temp_out, keepdims=keepdims)

                divisor = (
                    cp.asarray(_mean_count(a.shape, axis), dtype=np.int64)
                    if count is None else count
                )
                y = cp.divide(total, divisor, out=total, casting="unsafe")

                if out is None:
                    y = y.astype(result_dtype, copy=False)

        y = _write_output(out, y)
        destination = x if in_place and x_is_tensor else None
        result = _finish_output(y, op_device, requires_grad, aliases, destination=destination)

        if requires_grad and x_is_tensor and x.requires_grad:
            self.link(result, x, axis, keepdims, mask, count)

        return result

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
clip = clip_cls()
