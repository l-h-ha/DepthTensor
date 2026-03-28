"""
Neural Network operations
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['relu_forward_cpu']
@typing.overload
def relu_forward_cpu(arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
    """
    ReLU for float32
    """
@typing.overload
def relu_forward_cpu(arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
    """
    ReLU for float64
    """
