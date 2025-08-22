from ..DepthTensor import Tensor, differentiate, random

a = Tensor([1, 2, 3])
print(Tensor.where(a > 0, 1.0, 0.0))
