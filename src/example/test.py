from ..DepthTensor import Tensor

import numpy as np
import cupy as cp

x = cp.array(2)

a = Tensor(x)
a = Tensor(a)
a *= 2
print(a)
print(a.data.dtype)