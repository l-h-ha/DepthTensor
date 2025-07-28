from ..DTensor import Tensor

import numpy as np
import cupy as cp

a = Tensor(data=1, device="cpu")
b = 1
a += b
print(a)