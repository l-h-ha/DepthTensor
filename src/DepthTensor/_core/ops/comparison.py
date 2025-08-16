from typing import (
    Union, 
    Optional, 
    Tuple, 
    overload
)

from ...typing import (
    ArrayLike,
    TensorLike,
    DeviceLike
)

from ..exceptions import (
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

@overload
def where(
    condition: Union[ArrayLike, TensorLike],
    /
) -> Tuple[TensorLike, ...]: ...

@overload
def where(
    condition: Union[ArrayLike, TensorLike],
    x: Optional[Union[ArrayLike, TensorLike]],
    y: Optional[Union[ArrayLike, TensorLike]],
    /,
    *,
    device: DeviceLike = "cpu"
) -> TensorLike: ...

def where(
    condition: Union[ArrayLike, TensorLike],
    x: Optional[Union[ArrayLike, TensorLike]] = None,
    y: Optional[Union[ArrayLike, TensorLike]] = None,
    /,
    *,
    device: DeviceLike = "cpu"
) -> Union[Tuple[TensorLike, ...], TensorLike]:
    from ...tensor import Tensor
    #* One parameter overload
    if x is None and y is None:
        if isinstance(condition, TensorLike):
            data = condition.data
        else:
            if device == "cpu":
                data = np.asarray(condition)
            else:
                if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                data = cp.asarray(condition)
        if device == "cpu":
            result = np.where(data)
        else:
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data)
        return tuple([Tensor(array, device=device) for array in result])
    elif x is not None and y is not None:
        if isinstance(condition, TensorLike):
            data = condition.data
        else:
            if device == "cpu":
                data = np.asarray(condition)
            else:
                if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                data = cp.asarray(condition)
        if isinstance(x, TensorLike):
            x_data = x.data
        else:
            if device == "cpu":
                x_data = np.asarray(x)
            else:
                if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                x_data = cp.asarray(x)
        if isinstance(y, TensorLike):
            y_data = y.data
        else:
            if device == "cpu":
                y_data = np.asarray(y)
            else:
                if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                y_data = cp.asarray(y)
        if device == "cpu":
            result = np.where(data, x_data, y_data)
        else:
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data, x_data, y_data)
        return Tensor(result, device=device)
    else:
        raise ValueError("Both x and y parameters must be given.")

###
###
###

__all__ = [
    'where'
]