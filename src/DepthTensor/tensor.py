from __future__ import annotations
from typing import (
    Union, 
    Optional, 
    Any,
    Tuple, 
    Callable, 
    overload
)

from .typing import (
    ArrayLike, 
    DTypeLike,
    Order,
    DeviceLike, 
    ShapeLike, 
    ArrayLikeBool,
    Casting, 
    AxisShapeLike,
    OperandLike
)

from ._core import (
    xp_array_to_device,
    CuPyNotFound, 
    CUPY_NOT_FOUND_MSG,

    #* creation
    zeros_like, 
    ones_like,

    #* elementwise
    add, 
    subtract, 
    multiply, 
    matmul, 
    divide,
    negative, 
    power,
    clip, 
    sign, 
    abs,
    #* elementwise (exponents/logarithm)
    exp, 
    sqrt, 
    log, 
    square,

    #* diff (elementwise)
    add_diff, 
    subtract_diff, 
    multiply_diff, 
    matmul_diff, 
    divide_diff,
    power_diff,
    negative_diff, 
    sign_diff, 
    abs_diff,
    #* diff (exponents/logarithm)
    exp_diff, 
    sqrt_diff,
    log_diff,
    square_diff,
    

    #* comparison
    where, 
    equal, 
    not_equal, 
    greater, 
    greater_equal, 
    less, 
    less_equal,

    #* reduction
    max, 
    maximum, 
    sum
)

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None
_NoValue = object()

###
###
###

def _wrapper_2in_1out(y, diff_func, x1, x2, record_op):
    if record_op:
        return diff_func(y, x1, x2)
    return y

def _wrapper_1in_1out(y, diff_func, x, record_op):
    if record_op:
        return diff_func(y, x)
    return y

###
###
###

class Tensor():
    device: DeviceLike
    backward: Optional[Callable[[], None]]
    data: Union[np.ndarray, Any]

    def __init__(
        self,
        obj: Union[ArrayLike, Any, Tensor],
        /,
        *,
        dtype: Optional[DTypeLike] = None,
        device: DeviceLike = 'cpu',   
        prev: Tuple = (),
        requires_grad: bool = False,
        
        copy: bool = True,
        order: Order = 'K',
        subok: bool = False,
        ndmin: int = 0,
        blocking: bool = False
    ) -> None:
        #* Convert to xp.ndarray
        if isinstance(obj, np.ndarray):
            self.data = xp_array_to_device(obj, device)
            self.device = "cpu"
        else:
            if cp is not None:
                if isinstance(obj, cp.ndarray):
                    self.data = xp_array_to_device(obj, device)
                    self.device = "gpu"
                else:
                    if device == "gpu":
                        self.data = cp.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, blocking=blocking)
                        self.device = "gpu"
                    else:
                        self.data = np.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
                        self.device = "cpu"
            else:
                if device == "cpu":
                    self.data = np.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
                    self.device = "cpu"
                else:
                    raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        #* Convert to dtype (if provided)
        if dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype) 
        self.prev = prev
        self.requires_grad = requires_grad
        
        if device == "cpu":
            self.grad = np.zeros_like(self.data)
        else:
            if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            self.grad = cp.zeros_like(self.data)
        self.backward = None

    ###
    ###
    ###

    def get_device(self) -> DeviceLike:
        return self.device
    def is_device(self, device: DeviceLike) -> bool:
        return self.device == device
    def is_cpu(self) -> bool:
        return self.device == "cpu"
    def is_gpu(self) -> bool:
        return self.device == "gpu"

    ###
    ### Property
    ###

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype
    
    @property
    def shape(self) -> ShapeLike:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def size(self) -> int:
        return self.data.size
    
    ###
    ### Creation
    ###

    @staticmethod
    def zeros_like(
        a: OperandLike,
        /,
        *,
        device: DeviceLike = "cpu",
        dtype: Optional[DTypeLike] = None, 
        order: Order = 'K', 
        subok: bool = True,
        shape: Optional[AxisShapeLike] = None
    ) -> Tensor:
        return zeros_like(a, device=device, dtype=dtype, order=order, subok=subok, shape=shape)
    
    @staticmethod
    def ones_like(
        a: OperandLike,
        /,
        *, 
        device: DeviceLike = "cpu",
        dtype: Optional[DTypeLike] = None, 
        order: Order = 'K', 
        subok: bool = True,
        shape: Optional[AxisShapeLike] = None
    ) -> Tensor:
        return ones_like(a, device=device, dtype=dtype, order=order, subok=subok, shape=shape)
    
    ###
    ### Element-wise
    ###

    @staticmethod
    def add(
        x1: OperandLike, 
        x2: OperandLike, 
        /,
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_2in_1out(
            add(x1, x2, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            add_diff, x1, x2, record_op
        )
    
    @staticmethod
    def subtract(
        x1: OperandLike, 
        x2: OperandLike, 
        /,
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_2in_1out(
            subtract(x1, x2, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            subtract_diff, x1, x2, record_op
        )

    @staticmethod
    def multiply(
        x1: OperandLike, 
        x2: OperandLike, 
        /,
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_2in_1out(
            multiply(x1, x2, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            multiply_diff, x1, x2, record_op
        )
    
    @staticmethod
    def matmul(
        x1: OperandLike, 
        x2: OperandLike, 
        /,
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_2in_1out(
            matmul(x1, x2, out=out, device=device, in_place=in_place, casting=casting, order=order, dtype=dtype, subok=subok),
            matmul_diff, x1, x2, record_op
        )
    
    @staticmethod
    def divide(
        x1: OperandLike, 
        x2: OperandLike, 
        /,
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_2in_1out(
            divide(x1, x2, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            divide_diff, x1, x2, record_op
        )
    
    @staticmethod
    def power(
        x1: OperandLike, 
        x2: OperandLike, 
        /,
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_2in_1out(
            power(x1, x2, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            power_diff, x1, x2, record_op
        )
    
    @staticmethod
    def negative(
        x: OperandLike,
        /,
        out: Optional[Union[np.ndarray, Any]] = None, 
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            negative(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            negative_diff, x, record_op
        )
    
    @staticmethod
    def clip(
        a: OperandLike,
        a_min: OperandLike,
        a_max: OperandLike,
        /,
        out: Optional[ArrayLike] = None,
        *,
        device: DeviceLike = "cpu",
        where: Union[bool, ArrayLikeBool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return clip(a, a_min, a_max, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    
    @staticmethod
    def sign(
        x: OperandLike,
        /,
        out: Optional[Union[np.ndarray, Any]] = None, 
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            sign(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            sign_diff, x, record_op
        )
    
    @staticmethod
    def abs(
        x: OperandLike,
        /,
        out: Optional[Union[np.ndarray, Any]] = None, 
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[ArrayLikeBool, bool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            abs(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            abs_diff, x, record_op
        )

    ###
    ### Exponents/Logarithms
    ###

    @staticmethod
    def exp(
        x: OperandLike, 
        /, 
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[bool, ArrayLikeBool] = True, 
        casting: Casting = 'same_kind',
        order: Order = 'K', 
        dtype: Optional[DTypeLike] = None, 
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            exp(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            exp_diff, x, record_op
        )
    
    @staticmethod
    def sqrt(
        x: OperandLike, 
        /, 
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[bool, ArrayLikeBool] = True, 
        casting: Casting = 'same_kind',
        order: Order = 'K', 
        dtype: Optional[DTypeLike] = None, 
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            sqrt(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            sqrt_diff, x, record_op
        )
    
    @staticmethod
    def log(
        x: OperandLike, 
        /, 
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[bool, ArrayLikeBool] = True, 
        casting: Casting = 'same_kind',
        order: Order = 'K', 
        dtype: Optional[DTypeLike] = None, 
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            log(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            log_diff, x, record_op
        )
    
    @staticmethod
    def square(
        x: OperandLike, 
        /, 
        out: Optional[Union[np.ndarray, Any]] = None,
        *,
        device: DeviceLike = "cpu",
        in_place: bool = False,
        record_op: bool = True,

        where: Union[bool, ArrayLikeBool] = True, 
        casting: Casting = 'same_kind',
        order: Order = 'K', 
        dtype: Optional[DTypeLike] = None, 
        subok: bool = True
    ) -> Tensor:
        return _wrapper_1in_1out(
            square(x, out=out, device=device, in_place=in_place, where=where, casting=casting, order=order, dtype=dtype, subok=subok),
            square_diff, x, record_op
        )
    
    ###
    ### Comparison
    ###

    @staticmethod
    @overload
    def where(
        condition: OperandLike,
        /,
        *,
        device: DeviceLike = "cpu",
    ) -> Tuple[Tensor, ...]: ...

    @staticmethod
    @overload
    def where(
        condition: OperandLike,
        x: Optional[OperandLike],
        y: Optional[OperandLike],
        /,
        *,
        device: DeviceLike = "cpu",
    ) -> Tensor: ...

    @staticmethod
    def where(
        condition: OperandLike,
        x: Optional[OperandLike] = None,
        y: Optional[OperandLike] = None,
        /,
        *,
        device: DeviceLike = "cpu",
    ) -> Union[Tuple[Tensor, ...], Tensor]:
        device = "cpu"
        if isinstance(condition, Tensor):
            device = condition.device
        if x and isinstance(x, Tensor):
            if x.device != device:
                raise RuntimeError("Arguments, as tensors, must have the same device.")
        if y and isinstance(y, Tensor):
            if y.device != device:
                raise RuntimeError("Arguments, as tensors, must have the same device.")
        return where(condition, x, y, device=device)
    
    @staticmethod
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
    ) -> Tensor:
        return equal(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    
    @staticmethod
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
    ) -> Tensor:
        return not_equal(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    
    @staticmethod
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
    ) -> Tensor:
        return greater(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    @staticmethod
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
    ) -> Tensor:
        return greater_equal(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    @staticmethod
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
    ) -> Tensor:
        return less(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    @staticmethod
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
    ) -> Tensor:
        return less_equal(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    ###
    ### Reduction
    ###

    @staticmethod
    def sum(
        a: OperandLike,
        /,
        *,
        device: DeviceLike = "cpu",
        axis: Optional[AxisShapeLike] = None,
        dtype: Optional[DTypeLike] = None,
        out: Optional[Union[np.ndarray, Any]] = None,
        keepdims: bool = True,
        initial: Any = _NoValue,
        where: Union[bool, ArrayLikeBool] = True
    ) -> Tensor:
        return sum(a, axis=axis, device=device, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)

    @staticmethod
    def max(
        a: OperandLike,
        /,
        *,
        device: DeviceLike = "cpu",
        axis: Optional[ShapeLike] = None, 
        out: Optional[Union[np.ndarray, Any]] = None, 
        keepdims: bool = False, 
        initial: Any = _NoValue, 
        where: Union[bool, ArrayLikeBool] = True
    ) -> Tensor:
        return max(a, axis=axis, device=device, out=out, keepdims=keepdims, initial=initial, where=where)
    
    @staticmethod
    def maximum(
        x1: OperandLike,
        x2: OperandLike,
        /,
        out: Optional[np.ndarray] = None,
        *,
        device: DeviceLike = "cpu",
        where: Union[bool, ArrayLikeBool] = True,
        casting: Casting = 'same_kind',
        order: Order = 'K',
        dtype: Optional[DTypeLike] = None,
        subok: bool = True
    ) -> Tensor:
        return maximum(x1, x2, out=out, device=device, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    ###
    ### Dunder Operations
    ###

    def __add__(self, t: OperandLike) -> Tensor:
        return Tensor.add(self, t)
    def __radd__(self, t: OperandLike) -> Tensor:
        return Tensor.add(t, self)
    def __iadd__(self, t: OperandLike) -> Tensor:
        return Tensor.add(self, t, in_place=True)
    
    def __sub__(self, t: OperandLike) -> Tensor:
        return Tensor.subtract(self, t)
    def __rsub__(self, t: OperandLike) -> Tensor:
        return Tensor.subtract(t, self)
    def __isub__(self, t: OperandLike) -> Tensor:
        return Tensor.subtract(self, t, in_place=True)
    
    def __mul__(self, t: OperandLike) -> Tensor:
        return Tensor.multiply(self, t)
    def __rmul__(self, t: OperandLike) -> Tensor:
        return Tensor.multiply(t, self)
    def __imul__(self, t: OperandLike) -> Tensor:
        return Tensor.multiply(self, t, in_place=True)
    
    def __matmul__(self, t: OperandLike) -> Tensor:
        return Tensor.matmul(self, t)
    def __rmatmul__(self, t: OperandLike) -> Tensor:
        return Tensor.matmul(t, self)
    def __imatmul__(self, t: OperandLike) -> Tensor:
        return Tensor.matmul(self, t, in_place=True)
    
    def __truediv__(self, t: OperandLike) -> Tensor:
        return Tensor.divide(self, t)
    def __rtruediv__(self, t: OperandLike) -> Tensor:
        return Tensor.divide(t, self)
    def __itruediv__(self, t: OperandLike) -> Tensor:
        return Tensor.divide(self, t, in_place=True)
    
    def __pow__(self, t: OperandLike) -> Tensor:
        return Tensor.power(self, t)
    def __ipow__(self, t: OperandLike) -> Tensor:
        return Tensor.power(self, t, in_place=True)

    ###
    ### Unary
    ###

    def __eq__(self, value: Any) -> Tensor: # type: ignore[override]
        return equal(self, value)
    
    def __ne__(self, value: Any) -> Tensor: # type: ignore[override]
        return not_equal(self, value)
    
    def __gt__(self, value: Any) -> Tensor: # type: ignore[override]
        return greater(self, value)

    def __ge__(self, value: Any) -> Tensor: # type: ignore[override]
        return greater_equal(self, value)

    def __lt__(self, value: Any) -> Tensor: # type: ignore[override]
        return less(self, value)

    def __le__(self, value: Any) -> Tensor: # type: ignore[override]
        return less_equal(self, value)

    def __neg__(self) -> Tensor:
        return Tensor.negative(self)

    ###
    ### Misc dunder
    ###

    def __repr__(self) -> str:
        return f'Tensor({self.data}, device={self.device})'
    
    def __hash__(self) -> int:
        return id(self)