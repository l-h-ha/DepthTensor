from .. import DepthTensor as DTensor

import numpy as np
import cupy as cp

arr = cp.array(1.0)
a = DTensor.Tensor(arr, requires_grad=True)
print(type(a.grad))
