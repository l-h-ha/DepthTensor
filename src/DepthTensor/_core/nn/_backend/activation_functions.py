from ....typing import (
    TensorType,
    TensorLike,
    TensorData,
    Device,
    DTypeLike,
)

from ...ops.function import Function
from ...exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG
from ...utils import unbroadcast_tensordata_to_shape, get_device, to_tensordata

from .... import _ext as ext

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None


class relu_cls(Function):
    def link(self, y: TensorType, x: TensorLike) -> None:
        def callback(y_grad, x_shape, device, x_data) -> TensorData:
            grad = y_grad * (x_data > 0)
            return unbroadcast_tensordata_to_shape(grad, x_shape, device)

        from ....tensor import Tensor

        def grad_fn() -> None:
            if y.grad is None:
                y.zero_grad()

            result_grad = y.grad
            _x = to_tensordata(x)
            if isinstance(x, Tensor) and x.requires_grad:
                if x.grad is None:
                    x.zero_grad()
                x.grad += callback(result_grad, x.shape, x.device, _x)

        if isinstance(x, Tensor) and x.requires_grad:
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
        Applies the rectified linear unit function element-wise.

        Parameters
        ----------
        x : TensorLike
            The input data.
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        requires_grad : bool, optional
            Whether to link the result to the computation graph for autodiff.

        Returns
        -------
        TensorType
            The result of the ReLU operation.
        """

        from ....tensor import Tensor

        x_is_tensor = isinstance(x, Tensor)
        if x_is_tensor:
            x = x.contiguous()

        op_device = device if device is not None else get_device(x)
        a = to_tensordata(x, op_device)

        if op_device == "cpu":
            # TODO: data should be type-agnostic, not being dependant on float32
            if a.dtype != np.float32:
                a = a.astype(np.float32)

            y_data = a if in_place else a.copy()
            ext.nn.relu_forward_cpu(y_data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            y_data = cp.maximum(0, a, out=out)

        req_grad = requires_grad or (x_is_tensor and x.requires_grad)
        if in_place and x_is_tensor:
            if x.requires_grad:
                raise RuntimeError(
                    "In-place operations are forbidden on differentiable tensors."
                )
            x.data = y_data
            return x

        y = Tensor._fast_init(y_data, op_device, requires_grad=req_grad)
        if req_grad:
            self.link(y, x)
        return y


relu = relu_cls()
