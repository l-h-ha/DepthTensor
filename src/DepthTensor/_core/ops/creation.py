from typing import Optional

from ...typing import TensorLike, DTypeLike, Order, AxisLike, OperandLike, DeviceLike

from ..exceptions import CuPyNotFound, CUPY_NOT_FOUND_MSG

from ..utils import to_xp_array, get_device

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


def zeros_like(
    a: OperandLike,
    /,
    *,
    device: Optional[DeviceLike] = None,
    requires_grad: bool = False,
    dtype: Optional[DTypeLike] = None,
    order: Order = "K",
    subok: bool = True,
    shape: Optional[AxisLike] = None,
) -> TensorLike:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device
    a = to_xp_array(a)
    if device_op == "cpu":
        return Tensor(
            np.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape),
            device=device_op,
            requires_grad=requires_grad,
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(
            cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape),
            device=device_op,
            requires_grad=requires_grad,
        )


def ones_like(
    a: OperandLike,
    /,
    *,
    device: Optional[DeviceLike] = None,
    requires_grad: bool = False,
    dtype: Optional[DTypeLike] = None,
    order: Order = "K",
    subok: bool = True,
    shape: Optional[AxisLike] = None,
) -> TensorLike:
    from ...tensor import Tensor

    if device is None:
        device_op = get_device(a)
    else:
        device_op = device
    a = to_xp_array(a)
    if device_op == "cpu":
        return Tensor(
            np.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape),
            device=device_op,
            requires_grad=requires_grad,
        )
    else:
        if cp is None:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return Tensor(
            cp.zeros_like(a, dtype=dtype, order=order, subok=None, shape=shape),
            device=device_op,
            requires_grad=requires_grad,
        )


###
###
###

__all__ = ["zeros_like", "ones_like"]
