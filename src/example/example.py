from ..DepthTensor import (Tensor, differentiate, random, create_1in_1out, Op_2in_1out_Protocol, Func_2in_1out_Protocol, OperandLike, Diff_2in_1out_Protocol, ArrayLike, DeviceLike, CuPyNotFound, CUPY_NOT_FOUND_MSG)

import numpy as np
try:
    import cupy as cp
except:
    cp = None

def op(x: ArrayLike, device: DeviceLike = "cpu", **kwds) -> ArrayLike:
    if device == "cpu":
        return np.square(x) 
    else:
        if cp is None: raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
        return cp.square(x)

def diff(result: Tensor, x: ArrayLike):
    def x_diff() -> ArrayLike:
        return 2 * x
    return x_diff

func = create_1in_1out(op, diff)
a = Tensor(6.0, device="gpu", requires_grad=True)
b = func(a, device="gpu")
print(b)
differentiate(b)
print(a.grad)