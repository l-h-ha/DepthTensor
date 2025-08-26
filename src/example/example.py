from ..DepthTensor import Tensor, differentiate, random

a = Tensor([1, 2, 3], requires_grad=True)
b = a**2**2 + 1
print(a.requires_grad)
print(b.requires_grad)
differentiate(b)
print(a.grad)
print(b.prev)