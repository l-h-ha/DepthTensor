from ..DepthTensor import (Tensor, differentiate, random, create_1in_1out, Op_2in_1out_Protocol, Func_2in_1out_Protocol, OperandLike, Diff_2in_1out_Protocol, NDArrayLike, DeviceLike, CuPyNotFound, CUPY_NOT_FOUND_MSG)

import numpy as np
try:
    import cupy as cp
except:
    cp = None

def op(x: NDArrayLike, device: DeviceLike = "cpu", **kwds) -> NDArrayLike:
    print(type(x))
    return Tensor.add(x, x)

def diff(result: Tensor, x: NDArrayLike):
    def x_diff() -> NDArrayLike:
        return 1
    return x_diff

func = create_1in_1out(op, diff)
a = Tensor(6.0, device="gpu", requires_grad=True)
b = func(a, device="gpu")
print(b)
print(b.requires_grad)
print(b.prev)
differentiate(b)
print(a.grad)