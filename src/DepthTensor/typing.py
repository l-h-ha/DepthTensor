from typing import (
    TypeAlias,
    Union,
    Literal,
    Tuple,
    List,
    TYPE_CHECKING,
    Protocol,
    Any,
    Callable,
    Optional,
)

if TYPE_CHECKING:
    from .tensor import Tensor
TensorLike: TypeAlias = "Tensor"

import numpy as np
import numpy.typing as npt

DeviceLike: TypeAlias = Literal["cpu", "gpu"]
ScalarLike: TypeAlias = Union[int, float, bool]
ShapeLike: TypeAlias = Tuple[int, ...]
AxisLike: TypeAlias = Union[int, ShapeLike]
Order: TypeAlias = Literal["K", "A", "C", "F"]
Casting: TypeAlias = Literal["no", "equiv", "safe", "same_kind", "unsafe"]

DTypeLike: TypeAlias = npt.DTypeLike
floating: TypeAlias = np.floating
float16: TypeAlias = np.float16
float32: TypeAlias = np.float32
float64: TypeAlias = np.float64
integer: TypeAlias = np.integer
int8: TypeAlias = np.int8
int16: TypeAlias = np.int16
int32: TypeAlias = np.int32
int64: TypeAlias = np.int64
double: TypeAlias = np.double

NDArrayLike: TypeAlias = Union[npt.NDArray[np.number], Any]
NDArrayLikeBool: TypeAlias = Union[npt.NDArray[np.bool_], Any]
OperandLike: TypeAlias = Union[
    ScalarLike, NDArrayLike, NDArrayLikeBool, TensorLike, List, Tuple
]


class cfunc_2in_1out_pro(Protocol):
    def __call__(
        self,
        x1: OperandLike,
        x2: OperandLike,
        *,
        device: Optional[DeviceLike] = None,
        requires_grad: bool = True,
        **kwds: Any
    ) -> TensorLike: ...


class cop_2in_1out_pro(Protocol):
    def __call__(
        self, x1: NDArrayLike, x2: NDArrayLike, *, device: DeviceLike, **kwds: Any
    ) -> OperandLike: ...


class cdiff_2in_1out_pro(Protocol):
    def __call__(
        self, result: TensorLike, x1: NDArrayLike, x2: NDArrayLike, **kwds: Any
    ) -> Tuple[Callable[[], NDArrayLike], Callable[[], NDArrayLike]]: ...


class cfunc_1in_1out_pro(Protocol):
    def __call__(
        self,
        x: OperandLike,
        *,
        device: Optional[DeviceLike] = None,
        requires_grad: bool = True,
        **kwds: Any
    ) -> TensorLike: ...


class cop_1in_1out_pro(Protocol):
    def __call__(
        self, x: NDArrayLike, *, device: DeviceLike, **kwds: Any
    ) -> OperandLike: ...


class cdiff_1in_1out_pro(Protocol):
    def __call__(
        self, result: TensorLike, x: NDArrayLike, **kwds: Any
    ) -> Callable[[], NDArrayLike]: ...


__all__ = [
    "DTypeLike",
    "floating",
    "float16",
    "float32",
    "float64",
    "integer",
    "int8",
    "int16",
    "int16",
    "int64",
    "double",
    "DeviceLike",
    "Order",
    "AxisLike",
    "ScalarLike",
    "NDArrayLike",
    "NDArrayLikeBool",
    "ShapeLike",
    "TensorLike",
    "OperandLike",
    "cop_2in_1out_pro",
    "cdiff_2in_1out_pro",
    "cfunc_2in_1out_pro",
    "cop_1in_1out_pro",
    "cdiff_1in_1out_pro",
    "cfunc_1in_1out_pro",
]
