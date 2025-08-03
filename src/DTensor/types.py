from typing import TypeAlias
from numpy.typing import DTypeLike as DTypeLike_
import numpy as np

DTypeLike: TypeAlias = DTypeLike_

floating: TypeAlias = np.floating
float16: TypeAlias = np.float16
float32: TypeAlias = np.float32
float64: TypeAlias = np.float64
float96: TypeAlias = np.float96
float128: TypeAlias = np.float128

integer: TypeAlias = np.integer
int8: TypeAlias = np.int8
int16: TypeAlias = np.int16
int32: TypeAlias = np.int32
int64: TypeAlias = np.int64

double: TypeAlias = np.double

__all__ = [
    'DTypeLike', 
    'floating', 'float16', 'float32', 'float64', 'float96', 'float128',
    'integer', 'int8', 'int16', 'int16', 'int64',
    'double'
    ]