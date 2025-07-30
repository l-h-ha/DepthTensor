from ..DTensor import Tensor

import numpy as np
import cupy as cp

a = Tensor(data=[123,132,123,13,542], device="cpu", requires_grad=True)
b = Tensor(data=[123,132,123,13,542], device="cpu", requires_grad=True)
c = (a.square() @ b).mean()
print(c, c.prev)