from src import DepthTensor as DTensor
import cupy as cp
import numpy as np

arr_1 = np.ones(shape=(10000, 5000), dtype=np.float32)
arr_2 = np.ones(shape=(5000, 10000), dtype=np.float32)
a = DTensor.Tensor(arr_1, requires_grad=True)
b = DTensor.Tensor(arr_2, requires_grad=True)
c = a @ b
DTensor.differentiate(c)
