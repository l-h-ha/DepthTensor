from typing import TypeAlias, Union, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor
TensorLike: TypeAlias = 'Tensor'

import numpy as np
import numpy.typing as npt

DeviceLike: TypeAlias = Literal["cpu", "gpu"]
ScalarLike: TypeAlias = Union[int, float, bool, complex]
AxisLike: TypeAlias = Union[int, Tuple[int, ...]]
ShapeLike: TypeAlias = Tuple[int, ...]
AxisShapeLike: TypeAlias = Union[int, ShapeLike]
Order: TypeAlias = Literal['K', 'A', 'C', 'F']
Casting: TypeAlias = Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe']

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

ArrayLike = npt.ArrayLike
ArrayLikeBool = npt.NDArray[np.bool_]
OperandLike: TypeAlias = Union[ArrayLike, TensorLike]

__all__ = [
    'DTypeLike', 
    'floating', 'float16', 'float32', 'float64',
    'integer', 'int8', 'int16', 'int16', 'int64',
    'double',
    'DeviceLike', 'AxisShapeLike', 'Order', 'AxisLike', 'ScalarLike', 'ArrayLike', 'ArrayLikeBool', 'ShapeLike',
    'TensorLike', 'OperandLike'
    ]