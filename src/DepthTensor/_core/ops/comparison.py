from typing import (
    Union, 
    Optional, 
    Tuple, 
    overload
)

from ...typing import (
    TensorLike,
    DeviceLike,
    ArrayLikeBool,
    Casting,
    Order,
    OperandLike
)

from ..exceptions import (
    CuPyNotFound, CUPY_NOT_FOUND_MSG,
    DeviceMismatch, DEVICE_MISMATCH_MSG,
)

from ..utils import to_xp_array

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
    condition: OperandLike,
    /,
    *,
    device: DeviceLike = "cpu"
) -> Tuple[TensorLike, ...]: ...

@overload
def where(
    condition: OperandLike,
    x: Optional[OperandLike],
    y: Optional[OperandLike],
    /,
    *,
    device: DeviceLike = "cpu"
) -> TensorLike: ...

def where(
    condition: OperandLike,
    x: Optional[OperandLike] = None,
    y: Optional[OperandLike] = None,
    /,
    *,
    device: DeviceLike = "cpu"
) -> Union[Tuple[TensorLike, ...], TensorLike]:
    from ...tensor import Tensor
    #* One parameter overload
    if (x is None) and (y is None):
        data = to_xp_array(condition, device=device)
        if device == "cpu":
            result = np.where(data)
        else:
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            result = cp.where(data)
        return tuple([Tensor(array, device=device) for array in result])
    elif x is not None and y is not None:
        data = to_xp_array(condition, device=device)
        x_data = to_xp_array(x, device=device)
        y_data = to_xp_array(y, device=device)
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

def wrapper_2in_1out(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    func_name: str,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    from ...tensor import Tensor
    is_tensor_op = False
    if isinstance(x1, Tensor) and isinstance(x2, Tensor):
        if not x1.is_device(x2.device):
            raise DeviceMismatch(DEVICE_MISMATCH_MSG)
        else:
            is_tensor_op = True
    #* Either tensor-tensor op or array-array op.

    #* If its tensor-tensor op, device is decided by x1.device, else, its by the device argument.
    if is_tensor_op and isinstance(x1, Tensor):
        device_op = x1.device
    else:
        device_op = device

    x1, x2 = to_xp_array(x1, device=device_op), to_xp_array(x2, device=device_op)
    if device_op == "cpu":
        y = getattr(np, func_name)(x1, x2, out=out, dtype=dtype, where=where, casting=casting, order=order, subok=subok)
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        y = getattr(cp, func_name)(x1, x2, out=out, dtype=dtype, casting=casting)
    return Tensor(y, device=device_op)

def equal(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return wrapper_2in_1out(x1, x2, out=out, func_name="equal", device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def not_equal(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return wrapper_2in_1out(x1, x2, out=out, func_name="not_equal", device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def greater(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return wrapper_2in_1out(x1, x2, out=out, func_name="greater", device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def greater_equal(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return wrapper_2in_1out(x1, x2, out=out, func_name="greater_equal", device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def less(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return wrapper_2in_1out(x1, x2, out=out, func_name="less", device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

def less_equal(
    x1: OperandLike,
    x2: OperandLike,
    /,
    out: Optional[ArrayLikeBool] = None,
    *,
    device: DeviceLike = "cpu",
    where: Union[bool, ArrayLikeBool] = True,
    casting: Casting = 'same_kind',
    order: Order = 'K',
    dtype: None = None,
    subok: bool = True
) -> TensorLike:
    return wrapper_2in_1out(x1, x2, out=out, func_name="less_equal", device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

###
###
###

__all__ = [
    'where',
    'equal', 
    'not_equal', 
    'greater', 
    'greater_equal', 
    'less', 
    'less_equal'
]