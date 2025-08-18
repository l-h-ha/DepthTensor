from typing import (
    Union, 
    Optional, 
    Tuple, 
    overload,
    Callable
)

from ...typing import (
    ArrayLike,
    TensorLike,
    DeviceLike,
    ArrayLikeBool,
    Casting,
    Order
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG,
    DeviceMismatch, DEVICE_MISMATCH_MSG
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

def comparison_ufunc(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    func_name: str,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    from ...tensor import Tensor
    if not x1.is_device(x2.device): raise DeviceMismatch(DEVICE_MISMATCH_MSG)
    if x1.is_cpu():
        y = getattr(np, func_name)(x1.data, x2.data, out=out, dtype=dtype, where=where, casting=casting, order=order, subok=subok)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = getattr(cp, func_name)(x1.data, x2.data, out=out, dtype=dtype, casting=casting)
    return Tensor(y, device=x1.device)

def equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return comparison_ufunc(x1, x2, out=out, func_name="equal", where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def not_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return comparison_ufunc(x1, x2, out=out, func_name="not_equal", where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def greater(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return comparison_ufunc(x1, x2, out=out, func_name="greater", where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def greater_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return comparison_ufunc(x1, x2, out=out, func_name="greater_equal", where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def less(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return comparison_ufunc(x1, x2, out=out, func_name="less", where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def less_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return comparison_ufunc(x1, x2, out=out, func_name="less_equal", where=where, casting=casting, order=order, dtype=dtype, subok=subok)

###
###
###

__all__ = [
    'where',
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal'
]