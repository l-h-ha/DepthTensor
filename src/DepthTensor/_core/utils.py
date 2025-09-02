from typing import Any, Union, Tuple, Optional

from ..typing import DeviceLike, OperandLike, NDArrayLike

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


def xp_array_to_device(
    obj: Union[np.ndarray, Any], device: DeviceLike
) -> Union[np.ndarray, Any]:
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


def sum_to_shape(result: Any, target_shape: Tuple, device: DeviceLike) -> Any:
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


def to_xp_array(a: OperandLike, device: Optional[DeviceLike] = None) -> NDArrayLike:
    """
    Convert data to numpy.ndarray/cp.ndarray.
    """
    from ..tensor import Tensor

    if isinstance(a, Tensor):
        y = a.data
    else:
        y = a
    if device is not None:
        if isinstance(y, np.ndarray):
            if device == "cpu":
                return y
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            return cp.array(y)
        else:
            if cp is not None and isinstance(y, cp.ndarray):
                if device == "gpu":
                    # ? In this scope, y is a cp.ndarray; however, pylance still thinks y is an OperandLike.
                    return y  # pyright: ignore[reportReturnType]
                return cp.asnumpy(y)
            else:
                if device == "cpu":
                    return np.array(y)
                else:
                    if cp is None:
                        raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                    return cp.array(y)
    else:
        if isinstance(y, np.ndarray):
            return y
        else:
            if cp is not None and isinstance(y, cp.ndarray):
                # ? In this scope, y is a cp.ndarray; however, pylance still thinks y is an OperandLike.
                return y  # pyright: ignore[reportReturnType]
            return np.array(y)


def get_device(a: OperandLike) -> DeviceLike:
    from ..tensor import Tensor

    if isinstance(a, Tensor):
        return a.device
    elif isinstance(a, (np.ndarray, np.floating, np.integer, np.bool)):
        return "cpu"
    elif cp is not None and isinstance(
        a, (cp.ndarray, cp.integer, cp.floating, cp.bool_)
    ):
        return "gpu"
    elif isinstance(a, (int, float, list, tuple, bool)):
        return "cpu"
    else:
        raise RuntimeError("Invalid argument type.")


def get_complement_device(device: DeviceLike) -> DeviceLike:
    if device == "cpu":
        return "gpu"
    else:
        return "cpu"


def get_two_operand_op_device(
    x1: OperandLike, x2: OperandLike, device: Optional[DeviceLike]
) -> DeviceLike:
    from ..tensor import Tensor

    if device is not None:
        return device

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
    "xp_array_to_device",
    "sum_to_shape",
    "to_xp_array",
    "get_two_operand_op_device",
]
