from .. import DepthTensor as DTensor

import numpy as np
import cupy as cp

a = DTensor.Tensor(0.1, requires_grad=True, device="gpu")
b = a**2
DTensor.differentiate(b)
print(b)
print(a.grad)
