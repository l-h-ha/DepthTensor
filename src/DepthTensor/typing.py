from typing import TypeAlias, Literal, TYPE_CHECKING, Protocol, Any, Callable, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .tensor import Tensor

###
###
###

Shape: TypeAlias = tuple[int, ...]
Axis: TypeAlias = int | Shape
Device: TypeAlias = Literal["cpu", "gpu"]
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

###
###
###

TensorType: TypeAlias = "Tensor"
TensorData: TypeAlias = npt.NDArray[Any] | Any
TensorDataBool: TypeAlias = npt.NDArray[np.bool_] | Any

ScalarLike: TypeAlias = int | float | bool | np.integer | np.floating | np.bool_
ArrayLike: TypeAlias = TensorData | ScalarLike | list | tuple
TensorLike: TypeAlias = Union[TensorType, ScalarLike | ArrayLike | TensorData]
