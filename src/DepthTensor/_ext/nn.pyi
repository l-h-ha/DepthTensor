"""
Native CPU neural network ops.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = ['relu_forward_cpu']

@typing.overload
def relu_forward_cpu(tensor: numpy.typing.NDArray[numpy.float32]) -> None:
    """
    Apply ReLU in place.

    Parameters
    ----------
    tensor : numpy.ndarray
        Writable C- or Fortran-contiguous float32 buffer.

    Raises
    ------
    RuntimeError
        The buffer is read-only or noncontiguous.
    """

@typing.overload
def relu_forward_cpu(tensor: numpy.typing.NDArray[numpy.float64]) -> None:
    """
    Apply ReLU in place.

    Parameters
    ----------
    tensor : numpy.ndarray
        Writable C- or Fortran-contiguous float64 buffer.

    Raises
    ------
    RuntimeError
        The buffer is read-only or noncontiguous.
    """
