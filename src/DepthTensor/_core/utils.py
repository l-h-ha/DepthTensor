from typing import Any

from ..typing import Device, TensorLike, TensorData

from .exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
)

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###


def tensordata_to_device(obj: TensorData, device: Device) -> TensorData:
    if isinstance(obj, np.ndarray):
        if device == "cpu":
            return obj
        # * gpu
        if cp is not None:
            return cp.array(obj)
        else:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
    else:
        if cp is not None and isinstance(obj, cp.ndarray):
            if device == "gpu":
                return obj
            # * cpu
            return cp.asnumpy(obj)
        else:
            raise RuntimeError(
                f"Expected argument 'obj' of type numpy.ndarray/cupy.ndarray, got: {type(obj)}"
            )


def unbroadcast_tensordata_to_shape(
    result: TensorData, target_shape: tuple, device: Device
) -> Any:
    """
    Reverses broadcasting to the un-broadcasted shape.

    When a variable was broadcasted in order to be compatible with the other, e.g. [1.0] + [1.0, 2.0, 3.0], differentiating
    the result w.r.t. the broadcasted variable such that the gradient matches the variable's gradient requires collapsing
    the result's shape down to the variable's.

    Let's say:
    Scalar A, vector B (1x3)

    C = A + B (A is broadcasted into a 1x3 vector)

    In order to calculate A's gradients, per the chain rule, we have to differentiate C w.r.t. A, which gives you a vector
    with the same shape as C's, even though the gradient's shape must match A's.

    Mathematically, since A influences every components of C, to get the gradient, we would have to sum every connections from
    A to C, which this function generalizes for every cases.
    """

    result_shape = result.shape
    if result_shape == target_shape:
        return result

    gained_dims = len(result_shape) - len(target_shape)
    if gained_dims > 0:
        # * Sum for gained dimensions.
        gained_axes = tuple([i for i in range(gained_dims)])

        if device == "cpu":
            result = np.sum(result, axis=gained_axes)
        elif device == "gpu":
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.sum(result, axis=gained_axes)

    # * Just collapsing the gained dimensions would not be enough, collapsing stretched dimensions is required too.
    stretched_axes = []
    for i, d in enumerate(target_shape):
        if result.ndim == 0:
            continue
        if d == 1 and result.shape[i] > 1:
            stretched_axes.append(i)
    if len(stretched_axes) > 0:
        if device == "cpu":
            result = np.sum(result, axis=tuple(stretched_axes), keepdims=True)
        elif device == "gpu":
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.sum(result, axis=tuple(stretched_axes), keepdims=True)
    return result


def to_tensordata(a: TensorLike, device: Device | None = None) -> TensorData:
    """
    Convert TensorLike to TensorData
    """
    from ..tensor import Tensor

    if not device:
        device = get_device(device)

    if isinstance(a, Tensor):
        y = a.data
    elif isinstance(a, np.ndarray) or (cp is not None and isinstance(a, cp.ndarray)):
        y = a
    else:
        if device == "cpu":
            y = np.array(a)
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            y = cp.array(a)

    # Convert if needed.
    if get_device(y) != device:
        return tensordata_to_device(y, device)
    return y


def get_device(a: TensorLike) -> Device:
    from ..tensor import Tensor

    if isinstance(a, Tensor):
        return a.device
    elif isinstance(a, (np.ndarray, np.number, np.bool_)):
        return "cpu"
    elif cp is not None and isinstance(a, (cp.ndarray, cp.number, cp.bool_)):
        return "gpu"
    elif isinstance(a, (int, float, list, tuple, bool)):
        return "cpu"
    else:
        raise RuntimeError(f"Invalid argument type: {type(a)}")


def get_complement_device(device: Device) -> Device:
    if device == "cpu":
        return "gpu"
    else:
        return "cpu"


def get_two_operand_op_device(
    x1: TensorLike, x2: TensorLike, device: Device | None
) -> Device:
    if device is not None:
        return device

    from ..tensor import Tensor

    if isinstance(x1, Tensor) and isinstance(x2, Tensor):
        if x1.device != x2.device:
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)
        op_device = x1.device
    else:
        if isinstance(x1, Tensor):
            op_device = x1.device
            if get_device(x2) != x1.device and not isinstance(
                x2,
                (
                    int,
                    float,
                    list,
                    tuple,
                ),  # * Universal datatypes regardless of devices
            ):
                # * This leaves x2 to be either np.array or cp.array.
                raise DeviceMismatch(
                    f"There is a incompatibility in datatypes between the two operands of types Tensor and {type(x2)}. Expected datatype of device: {x1.device} {"(CuPy arrays, GPU Tensors)" if x1.device == "gpu" else "(NumPy arrays, CPU Tensors, ints, floats, lists, tuples)"}, got datatype of type: {type(x2)}."
                )
        elif isinstance(x2, Tensor):
            op_device = x2.device
            if get_device(x1) != x2.device and not isinstance(
                x2,
                (
                    int,
                    float,
                    list,
                    tuple,
                ),  # * Universal datatypes regardless of devices
            ):
                # * This leaves x1 to be either np.array or cp.array.
                raise DeviceMismatch(
                    f"There is a incompatibility in datatypes between the two operands of types {type(x1)} and Tensor. Expected datatype of device: {x2.device} {"(CuPy arrays, GPU Tensors)" if x2.device == "gpu" else "(NumPy arrays, CPU Tensors, ints, floats, lists, tuples)"}, got datatype of type: {type(x1)}."
                )
        else:
            if isinstance(x1, np.ndarray) or isinstance(x2, np.ndarray):
                return "cpu"
            if cp is not None and (
                isinstance(x1, cp.ndarray) or isinstance(x2, cp.ndarray)
            ):
                return "gpu"
            return get_device(x1)

    return op_device


###
###
###

__all__ = [
    "tensordata_to_device",
    "unbroadcast_tensordata_to_shape",
    "to_tensordata",
    "get_two_operand_op_device",
]
