from contextlib import contextmanager
from typing import Iterator

from ....typing import (
    TensorType,
    TensorLike,
    TensorData,
    TensorDataBool,
    Device,
    DTypeLike,
)

from ...ops.function import Function, _ChainedBackward
from ...exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG
from ...output import _cuda_pointwise, _finish_output, _prepare_output, _write_output
from ...storage import _is_differentiable_result, _storage_aliases
from ...utils import (
    _accumulate_operand_grad,
    _check_dtype,
    _check_value_dtype,
    get_device,
    to_tensordata,
)

from .... import _ext as ext

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

class _InPlaceReluBackward(_ChainedBackward):
    def __init__(
        self, x: TensorType, positive_mask: TensorDataBool, previous_data: TensorData | None
    ) -> None:
        previous_backward = x.grad_fn

        if isinstance(previous_backward, _InPlaceReluBackward):
            # Copies may share the earlier callback.
            positive_mask = previous_backward.positive_mask & positive_mask
            previous_data = previous_backward.previous_data
            previous_backward = previous_backward.previous_backward

        self.x = x
        self.positive_mask = positive_mask
        self.previous_data = previous_data
        self.previous_backward = previous_backward

    @contextmanager
    def _use_saved_output(self) -> Iterator[None]:
        x = self.x
        if x.grad is None:
            x.zero_grad()

        xp = np if x.device == "cpu" else cp
        x.grad = xp.where(self.positive_mask, x.grad, x.grad.dtype.type(0))

        activated_data = x.data
        if self.previous_backward is not None:
            object.__setattr__(x, "data", self.previous_data)

        try:
            yield
        finally:
            object.__setattr__(x, "data", activated_data)

class relu_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        from ....tensor import Tensor

        if not isinstance(x, Tensor) or not x.requires_grad:
            return

        x_saved_version = x._version
        device = y.device
        xp = np if device == "cpu" else cp
        active = to_tensordata(x, device) > 0

        def grad_fn() -> None:
            if not x.requires_grad:
                return

            if x._version != x_saved_version:
                raise RuntimeError(
                    "Tensor modified by an in-place operation."
                )

            if y.grad is None:
                y.zero_grad()

            grad = xp.where(active, y.grad, y.grad.dtype.type(0))
            _accumulate_operand_grad(x, grad)

        y.prev = (x,)
        y.grad_fn = grad_fn

    def __call__(
        self,
        x: TensorLike,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        dtype: DTypeLike | None = None,
        requires_grad: bool = False,
    ) -> TensorType:
        """
        Apply ReLU.

        Parameters
        ----------
        x : TensorLike
            Input values.
        out : TensorData, optional
            Destination array.
        device : Device, optional
            Output device. Defaults to the input device.
        in_place : bool, optional
            Update tensor `x`. Tracked tensors stay on their device.
        dtype : DTypeLike, optional
            Validated only. ReLU computes in the input dtype.
        requires_grad : bool, optional
            Track the result if True or if x tracks gradients.

        Returns
        -------
        Tensor
            ReLU values.
        """
        from ....tensor import Tensor

        _check_dtype(dtype)
        _check_value_dtype(x)

        x_is_tensor = isinstance(x, Tensor)
        in_place_tensor = in_place and x_is_tensor
        op_device = device if device is not None else get_device(x)

        if in_place_tensor and x.requires_grad and op_device != x.device:
            raise RuntimeError("In-place device changes are forbidden on differentiable tensors.")

        req_grad = requires_grad or (x_is_tensor and x.requires_grad)
        out, out_aliases = _prepare_output(
            out, op_device, x, requires_grad=x.requires_grad if in_place_tensor else req_grad
        )

        if x_is_tensor and not in_place:
            x = x.contiguous()

        a = to_tensordata(x, op_device)

        if in_place_tensor and x.requires_grad:
            positive_mask = a > 0
            previous_data = x.data if x.grad_fn is not None else None
            if previous_data is not None and out is not None:
                previous_data = previous_data.copy()

        if op_device == "cpu":
            y_data = a if in_place and out is None else a.copy()
            if in_place_tensor and y_data is a and (
                x.grad_fn is not None or not a.flags.c_contiguous
            ):
                y_data = a.copy()

            if y_data is a:
                out_aliases = _storage_aliases(a)
                if any(_is_differentiable_result(tensor) for tensor in out_aliases):
                    raise RuntimeError(
                        "In-place ReLU would overwrite a differentiable result."
                    )

            if y_data.dtype in (np.float32, np.float64):
                ext.nn.relu_forward_cpu(y_data)
            else:
                np.maximum(y_data.dtype.type(0), y_data, out=y_data)

            y_data = _write_output(out, y_data, casting="same_kind")
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            y_data = _cuda_pointwise(cp.maximum, (a.dtype.type(0), a), out, None)

        if in_place_tensor:
            backward = (
                _InPlaceReluBackward(x, positive_mask, previous_data) if x.requires_grad else None
            )
            _finish_output(y_data, op_device, req_grad, out_aliases, destination=x)

            if backward is not None:
                x.grad_fn = backward
            return x

        y = _finish_output(y_data, op_device, req_grad, out_aliases)

        if req_grad:
            self.link(y, x)

        return y

relu = relu_cls()
