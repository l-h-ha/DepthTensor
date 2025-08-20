from typing import (
    Any, 
    Union, 
    Tuple,
    Optional
)

from ..typing import (
    DeviceLike, 
    TensorLike, 
    ArrayLike
)

from .exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG
)

import numpy as np
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

###
###
###

def xp_array_to_device(obj: Union[np.ndarray, Any], device: DeviceLike) -> Union[np.ndarray, Any]:
    if isinstance(obj, np.ndarray):
        if device == "cpu":
            return obj
        #* gpu
        if cp is not None:
            return cp.array(obj)
        else:
            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
    else:
        if cp is not None and isinstance(obj, cp.ndarray):
            if device == "gpu":
                return obj
            #* cpu
            return cp.asnumpy(obj)
        else:
            raise RuntimeError(f"Expected argument 'obj' of type numpy.ndarray/cupy.ndarray, got: {type(obj)}")

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
        #* Sum for gained dimensions.
        gained_axes = tuple([i for i in range(gained_dims)])
        
        if device == "cpu":
            result = np.sum(result, axis=gained_axes)
        elif device == "gpu":
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.sum(result, axis=gained_axes)

    #* Just collapsing the gained dimensions would not be enough, collapsing stretched dimensions is required too.
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
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.sum(result, axis=tuple(stretched_axes), keepdims=True)
    return result

def to_xp_array(a: Union[ArrayLike, TensorLike], device: Optional[DeviceLike] = None) -> ArrayLike:
    """
    Convert data to numpy.typing.ArrayLike
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
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            return cp.asarray(y)
        else:
            if cp is not None and isinstance(y, cp.ndarray):
                if device == "gpu":
                    return y
                return cp.asnumpy(y)
            else:
                return y
    return y

###
### 
###

__all__ = [
    'xp_array_to_device',
    'sum_to_shape',
    'to_xp_array'
]