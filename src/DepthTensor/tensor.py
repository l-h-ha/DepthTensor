from __future__ import annotations
from typing import Union, Optional, Any, Tuple, Callable, overload, Iterator, List

from .typing import (
    NDArrayLike,
    DTypeLike,
    Order,
    DeviceLike,
    ShapeLike,
    NDArrayLikeBool,
    Casting,
    OperandLike,
    AxisLike,
)

from ._core import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    # * elementwise
    add,
    subtract,
    multiply,
    matmul,
    divide,
    negative,
    power,
    clip,
    abs,
    # * diff (elementwise)
    add_diff,
    subtract_diff,
    multiply_diff,
    matmul_diff,
    divide_diff,
    power_diff,
    abs_diff,
    # * comparison
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
    # * reduction
    max,
    maximum,
    sum,
)

from ._core.utils import get_device, to_xp_array, xp_array_to_device

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None
_NoValue = object()

###
###
###


def _wrapper_2in_1out(
    y: Tensor,
    diff_func: Callable[[Tensor, OperandLike, OperandLike], Tensor],
    x1: OperandLike,
    x2: OperandLike,
    record_op: bool = True,
) -> Tensor:
    if record_op:
        return diff_func(y, x1, x2)
    return y


def _wrapper_1in_1out(
    y: Tensor,
    diff_func: Callable[[Tensor, OperandLike], Tensor],
    x: OperandLike,
    record_op: bool = True,
):
    if record_op:
        return diff_func(y, x)
    return y


allowed_dtype_kind = "uifb"

###
###
###


class Tensor:
    device: DeviceLike
    backward: Optional[Callable[[], None]]
    data: NDArrayLike

    def __init__(
        self,
        obj: OperandLike,
        /,
        *,
        dtype: Optional[DTypeLike] = None,
        device: Optional[DeviceLike] = None,
        prev: Tuple = (),
        requires_grad: bool = False,
        copy: bool = True,
        order: Order = "K",
        subok: bool = False,
        ndmin: int = 0,
        blocking: bool = False,
    ) -> None:
        if device is None:
            self.device = get_device(obj)
        else:
            self.device = device

        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in allowed_dtype_kind:
                self.data = xp_array_to_device(obj, self.device)
            else:
                raise TypeError("Expected a numerical NumPy array.")
        else:
            if cp is not None and isinstance(obj, cp.ndarray):
                if obj.dtype.kind in allowed_dtype_kind:  # type: ignore
                    self.data = xp_array_to_device(obj, self.device)
                else:
                    raise TypeError("Expected a numerical CuPy array.")
            elif isinstance(obj, Tensor):
                self.data = xp_array_to_device(obj.data, self.device)
            else:
                self.data = xp_array_to_device(
                    to_xp_array(obj, self.device), self.device
                )

        # TODO: Use all the parameters.

        # * Convert to dtype (if provided)
        if dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype)
        self.prev = prev
        self.requires_grad = requires_grad

        if self.device == "cpu":
            self.grad = np.zeros_like(self.data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            self.grad = cp.zeros_like(self.data)
        self.backward = None

    ###
    ###
    ###

    def copy(
        self,
        *,
        order: Order = "K",
        dtype: Optional[DTypeLike] = None,
        device: Optional[DeviceLike] = None,
        copy_prev: bool = False,
        copy_requires_grad: bool = False,
        copy_grad: bool = False,
    ) -> Tensor:
        t = Tensor(
            self.data.copy(order=order),
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            prev=self.prev if copy_prev else (),
            requires_grad=self.requires_grad if copy_requires_grad else False,
        )
        if copy_grad:
            t.grad = self.grad
        return t

    def make_differentiable(
        self, grad: Optional[Union[Tensor, np.ndarray, Any]] = None
    ) -> None:
        if not self.requires_grad:
            self.requires_grad = True

            if grad is None:
                if self.device == "cpu":
                    self.grad = np.zeros(self.shape)
                else:
                    if cp is None:
                        raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                    self.grad = cp.zeros(self.shape)
            else:
                if isinstance(grad, Tensor):
                    if grad.device != self.device:
                        raise RuntimeError(
                            "There is a mismatch in grad's device and tensor's device."
                        )
                    self.grad = grad.data
                elif isinstance(grad, np.ndarray):
                    if self.device == "gpu":
                        raise RuntimeError(
                            "Expected grad parameter to be a cupy.ndarray."
                        )
                    self.grad = grad
                elif cp is not None and isinstance(grad, cp.ndarray):
                    if self.device == "cpu":
                        raise RuntimeError(
                            "Expected grad parameter to be a numpy.ndarray."
                        )
                    self.grad = grad
                else:
                    raise RuntimeError(
                        "Expected grad parameter of specific types: Tensor, numpy.ndarray, cupy.ndarray."
                    )

    def to_device(
        self, device: DeviceLike, in_place: bool = False, clear_prev: bool = True
    ) -> Tensor:
        if device == self.device:
            if in_place:
                return self
            return self.copy()
        else:
            if in_place:
                self.device = device
                self.prev = () if clear_prev else self.prev
                self.data = xp_array_to_device(self.data, device=device)
                return self
            return self.copy(device=device)

    def get_device(self) -> DeviceLike:
        return self.device

    def is_device(self, device: DeviceLike) -> bool:
        return self.device == device

    def is_cpu(self) -> bool:
        return self.device == "cpu"

    def is_gpu(self) -> bool:
        return self.device == "gpu"

    ###
    ### Property
    ###

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def shape(self) -> ShapeLike:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    ###
    ### Element-wise
    ###

    def clip(
        self,
        a_min: OperandLike,
        a_max: OperandLike,
        /,
        out: Optional[NDArrayLike] = None,
        *,
        requires_grad: bool = False,
        device: DeviceLike = "cpu",
        where: Union[bool, NDArrayLikeBool] = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: Optional[DTypeLike] = None,
        subok: bool = True,
    ) -> Tensor:
        return clip(
            self,
            a_min,
            a_max,
            out=out,
            requires_grad=requires_grad,
            device=device,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    ###
    ### Reduction
    ###

    def sum(
        self,
        /,
        *,
        device: DeviceLike = "cpu",
        requires_grad: bool = False,
        axis: Optional[AxisLike] = None,
        dtype: Optional[DTypeLike] = None,
        out: Optional[NDArrayLike] = None,
        keepdims: bool = True,
        initial: Any = _NoValue,
        where: Union[bool, NDArrayLikeBool] = True,
    ) -> Tensor:
        return sum(
            self,
            axis=axis,
            device=device,
            requires_grad=requires_grad,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(
        self,
        /,
        *,
        device: DeviceLike = "cpu",
        requires_grad: bool = False,
        axis: Optional[AxisLike] = None,
        out: Optional[NDArrayLike] = None,
        keepdims: bool = False,
        initial: Any = _NoValue,
        where: Union[bool, NDArrayLikeBool] = True,
    ) -> Tensor:
        return max(
            self,
            axis=axis,
            device=device,
            requires_grad=requires_grad,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def maximum(
        self,
        x2: OperandLike,
        /,
        out: Optional[np.ndarray] = None,
        *,
        device: DeviceLike = "cpu",
        requires_grad: bool = False,
        where: Union[bool, NDArrayLikeBool] = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: Optional[DTypeLike] = None,
        subok: bool = True,
    ) -> Tensor:
        return maximum(
            self,
            x2,
            out=out,
            device=device,
            requires_grad=requires_grad,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    ###
    ### Dunder Operations
    ###

    def __add__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=add(self, t), diff_func=add_diff, x1=self, x2=t, record_op=True
        )

    def __radd__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=add(t, self), diff_func=add_diff, x1=t, x2=self, record_op=True
        )

    def __iadd__(self, t: OperandLike) -> Tensor:
        return add(self, t, in_place=True)

    def __sub__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=subtract(self, t), diff_func=subtract_diff, x1=self, x2=t, record_op=True
        )

    def __rsub__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=subtract(t, self), diff_func=subtract_diff, x1=t, x2=self, record_op=True
        )

    def __isub__(self, t: OperandLike) -> Tensor:
        return subtract(self, t, in_place=True)

    def __mul__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=multiply(self, t), diff_func=multiply_diff, x1=self, x2=t, record_op=True
        )

    def __rmul__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=multiply(t, self), diff_func=multiply_diff, x1=t, x2=self, record_op=True
        )

    def __imul__(self, t: OperandLike) -> Tensor:
        return multiply(self, t, in_place=True)

    def __matmul__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=matmul(self, t), diff_func=matmul_diff, x1=self, x2=t, record_op=True
        )

    def __rmatmul__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=matmul(t, self), diff_func=matmul_diff, x1=t, x2=self, record_op=True
        )

    def __imatmul__(self, t: OperandLike) -> Tensor:
        return matmul(self, t, in_place=True)

    def __truediv__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=divide(self, t), diff_func=divide_diff, x1=self, x2=t, record_op=True
        )

    def __rtruediv__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=divide(t, self), diff_func=divide_diff, x1=t, x2=self, record_op=True
        )

    def __itruediv__(self, t: OperandLike) -> Tensor:
        return divide(self, t, in_place=True)

    def __pow__(self, t: OperandLike) -> Tensor:
        return _wrapper_2in_1out(
            y=power(self, t), diff_func=power_diff, x1=self, x2=t, record_op=True
        )

    def __ipow__(self, t: OperandLike) -> Tensor:
        return power(self, t, in_place=True)

    ###
    ### Unary
    ###

    def __eq__(self, value: Any) -> Tensor:  # type: ignore[override]
        return equal(self, value)

    def __ne__(self, value: Any) -> Tensor:  # type: ignore[override]
        return not_equal(self, value)

    def __gt__(self, value: Any) -> Tensor:  # type: ignore[override]
        return greater(self, value)

    def __ge__(self, value: Any) -> Tensor:  # type: ignore[override]
        return greater_equal(self, value)

    def __lt__(self, value: Any) -> Tensor:  # type: ignore[override]
        return less(self, value)

    def __le__(self, value: Any) -> Tensor:  # type: ignore[override]
        return less_equal(self, value)

    def __neg__(self) -> Tensor:
        return negative(self)

    ###
    ### Misc dunder
    ###

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __setitem__(self, index, value) -> Any:
        self.data[index] = value

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, device={self.device})"

    def __hash__(self) -> int:
        return id(self)
