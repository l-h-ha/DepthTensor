from .. import DepthTensor as DTensor

import numpy as np
import cupy as cp

a = DTensor.add(cp.array(1.0), np.array(2.0))
print(a)
