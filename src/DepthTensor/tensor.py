from __future__ import annotations
from typing import Any, Callable, Iterator

from .typing import (
    TensorData,
    DTypeLike,
    Order,
    Device,
    Shape,
    TensorDataBool,
    Casting,
    TensorLike,
    Axis,
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
    mean,
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

from ._core.utils import get_device, to_tensordata, tensordata_to_device

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None
_NoValue = object()

###
###
###


def infer_differentiation_status(a: Tensor, b: TensorLike) -> bool:
    if isinstance(b, Tensor):
        return a.requires_grad or b.requires_grad
    return a.requires_grad


###
###
###


class Tensor:
    data: TensorData
    device: Device
    grad: TensorData | None
    backward: Callable[[], None] | None

    def __init__(
        self,
        obj: TensorLike,
        /,
        *,
        dtype: DTypeLike | None = None,
        device: Device | None = None,
        prev: tuple = (),
        requires_grad: bool = False,
        name: str = "",
    ) -> None:
        # Device init
        if device is None:
            self.device = get_device(obj)
        else:
            self.device = device

        # Data init
        if isinstance(obj, Tensor):
            self.data = obj.data
        elif isinstance(obj, np.ndarray):
            self.data = obj
        elif cp is not None and isinstance(obj, cp.ndarray):
            self.data = obj
        else:
            self.data = to_tensordata(obj, self.device)

        # Conversion
        if dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype)
        if get_device(self.data) != self.device:
            self.data = tensordata_to_device(self.data, self.device)

        # Other inits
        self.prev = prev
        self.requires_grad = requires_grad
        self.backward = None
        self.grad = None
        self.name = name

    def zero_grad(self) -> TensorData:
        if not self.requires_grad:
            raise RuntimeError(
                "Attempted to zero-gradient initialize an undifferentiable tensor."
            )
        if self.device == "cpu":
            grad = np.zeros_like(self.data)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            grad = cp.zeros_like(self.data)
        self.grad = grad
        return grad

    ###
    ###
    ###

    def copy(
        self,
        *,
        order: Order = "K",
        dtype: DTypeLike | None = None,
        device: Device | None = None,
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

    def make_differentiable(self, grad: Tensor | TensorData | None = None) -> None:
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
        self, device: Device, in_place: bool = False, clear_prev: bool = True
    ) -> Tensor:
        if device == self.device:
            if in_place:
                return self
            return self.copy()
        else:
            if in_place:
                if self.requires_grad:
                    raise RuntimeError(
                        "In-place operations (device switch) are forbidden on differentiable tensors."
                    )

                self.device = device
                self.prev = () if clear_prev else self.prev
                self.data = tensordata_to_device(self.data, device=device)
                return self
            return self.copy(device=device)

    def get_device(self) -> Device:
        return self.device

    def is_device(self, device: Device) -> bool:
        return self.device == device

    def is_cpu(self) -> bool:
        return self.device == "cpu"

    def is_gpu(self) -> bool:
        return self.device == "gpu"

    def set_name(self, name: str) -> Tensor:
        self.name = name
        return self

    def transpose(self, axes: Shape | None) -> Tensor:
        y = Tensor(
            self.data.transpose(axes),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        if y.requires_grad:

            def backward() -> None:
                if y.grad is None:
                    y.zero_grad()
                if self.grad is None:
                    self.zero_grad()

                if axes is None:
                    self.grad += y.grad.transpose(None)  # type: ignore
                else:
                    if self.is_cpu():
                        inverse_axes = np.argsort(axes)
                    else:
                        if cp is None:
                            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                        inverse_axes = cp.argsort(axes)
                    self.grad += y.grad.transpose(inverse_axes)  # type: ignore

            y.prev = (self,)
            y.backward = backward
        return y

    ###
    ### Property
    ###

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        self.item
        return self.data.size

    def item(self, **kwargs: Any) -> Any:
        return self.data.item(**kwargs)

    ###
    ### Element-wise
    ###

    def clip(
        self,
        a_min: TensorLike,
        a_max: TensorLike,
        /,
        out: TensorData | None = None,
        *,
        requires_grad: bool = False,
        device: Device = "cpu",
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
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

    def mean(
        self,
        /,
        axis: Axis | None = None,
        dtype: DTypeLike | None = None,
        out: TensorData | None = None,
        keepdims: bool = False,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
    ) -> Tensor:
        return mean(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            device=device,
            in_place=in_place,
            where=where,
            differentiate=self.requires_grad,
        )

    ###
    ### Reduction
    ###

    def sum(
        self,
        /,
        *,
        device: Device = "cpu",
        requires_grad: bool = False,
        axis: Axis | None = None,
        dtype: DTypeLike | None = None,
        out: TensorData | None = None,
        keepdims: bool = True,
        initial: Any = _NoValue,
        where: TensorDataBool | bool = True,
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
        device: Device = "cpu",
        requires_grad: bool = False,
        axis: Axis | None = None,
        out: TensorData | None = None,
        keepdims: bool = False,
        initial: Any = _NoValue,
        where: TensorDataBool | bool = True,
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
        x2: TensorLike,
        /,
        out: TensorData | None = None,
        *,
        device: Device = "cpu",
        requires_grad: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
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

    def __add__(self, t: TensorLike) -> Tensor:
        return add(self, t, differentiate=infer_differentiation_status(self, t))

    def __radd__(self, t: TensorLike) -> Tensor:
        return add(t, self, differentiate=infer_differentiation_status(self, t))

    def __iadd__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (add) forbidden on differentiable tensors."
            )
        return add(self, t, in_place=True)

    def __sub__(self, t: TensorLike) -> Tensor:
        return subtract(self, t, differentiate=infer_differentiation_status(self, t))

    def __rsub__(self, t: TensorLike) -> Tensor:
        return subtract(t, self, differentiate=infer_differentiation_status(self, t))

    def __isub__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (sub) are forbidden on differentiable tensors."
            )
        return subtract(self, t, in_place=True)

    def __mul__(self, t: TensorLike) -> Tensor:
        return multiply(self, t, differentiate=infer_differentiation_status(self, t))

    def __rmul__(self, t: TensorLike) -> Tensor:
        return multiply(t, self, differentiate=infer_differentiation_status(self, t))

    def __imul__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (mul) forbidden on differentiable tensors."
            )
        return multiply(self, t, in_place=True)

    def __matmul__(self, t: TensorLike) -> Tensor:
        return matmul(self, t, differentiate=infer_differentiation_status(self, t))

    def __rmatmul__(self, t: TensorLike) -> Tensor:
        return matmul(t, self, differentiate=infer_differentiation_status(self, t))

    def __imatmul__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (matmul) forbidden on differentiable tensors."
            )
        return matmul(self, t, in_place=True)

    def __truediv__(self, t: TensorLike) -> Tensor:
        return divide(self, t, differentiate=infer_differentiation_status(self, t))

    def __rtruediv__(self, t: TensorLike) -> Tensor:
        return divide(t, self, differentiate=infer_differentiation_status(self, t))

    def __itruediv__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (div) are forbidden on differentiable tensors."
            )
        return divide(self, t, in_place=True)

    def __pow__(self, t: TensorLike) -> Tensor:
        return power(self, t, differentiate=infer_differentiation_status(self, t))

    def __ipow__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (pow) are forbidden on differentiable tensors."
            )
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
        y = Tensor(
            self.data[index],
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        if y.requires_grad:

            def backward() -> None:
                if y.grad is None:
                    y.zero_grad()
                if self.grad is None:
                    self.zero_grad()
                self.grad[index] += y.grad  # type: ignore
                y.prev = (self,)

            y.backward = backward
        return y

    def __setitem__(self, index, value) -> Any:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (indexing) are forbidden on differentiable tensors."
            )
        self.data[index] = value

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, device={self.device}{f", name={self.name}" if self.name != "" else ""})"

    def __hash__(self) -> int:
        return id(self)
