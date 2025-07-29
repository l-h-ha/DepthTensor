from ..DTensor import Tensor

import numpy as np
import cupy as cp

a = Tensor(data=[123,132,123,13,542], device="cpu")
b = Tensor(data=[123,132,123,13,542], device="cpu")
c = a.square().dot(b)
print(c)